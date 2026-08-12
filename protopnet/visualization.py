import json
import logging
import pathlib
import random
import warnings
from typing import Union

import click
import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from . import datasets
from .utilities.general_utilities import find_high_activation_crop
from .utilities.project_utilities import custom_unravel_index, hash_func
from .utilities.visualization_utilities import indices_to_upsampled_boxes

log = logging.getLogger(__name__)


class KeyReturningDict(dict):
    """
    Dictionary that gives us default values for missing keys.
    Used to make some of the bootstrapping for the visualization less onerous.
    """

    def __missing__(self, key):
        return key


def cv2_overlay_heatmap(activation_0_to_1, ori_image):
    heatmap = cv2.applyColorMap(
        np.uint8(activation_0_to_1 * 255),
        cv2.COLORMAP_JET,
    )
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]
    overlayed_img = 0.5 * ori_image + 0.3 * heatmap
    return heatmap, overlayed_img


def extract_sample_id(sample_id: Union[torch.Tensor, int, str]) -> Union[int, str]:
    """
    Extract the sample id from the sample_id tensor respecting type safety of tensors, ints, and strs.
    """
    if isinstance(sample_id, torch.Tensor) and not torch.is_floating_point(sample_id):
        try:
            return int(sample_id.detach().cpu().item())
        except Exception:
            warnings.warn(
                f"Could not convert sample_id tensor {sample_id} to int. This may cause incorrect mapping of prototypes to samples."
            )
            return sample_id
    elif isinstance(sample_id, int) or isinstance(sample_id, str):
        return sample_id
    else:
        warnings.warn(
            f"Unknown sample_id type {type(sample_id)}. This may cause incorrect mapping of prototypes to samples."
        )
        return sample_id


def save_prototype_images_to_file(
    model,
    push_dataloader,
    save_loc,
    img_size,
    mean,
    std,
    normalize_for_fwd=None,
    box_color=(0, 255, 255),
    device="cpu",
):
    """
    Save prototype images, and images overlaid with heatmaps, to save_loc.
    """

    img_height, img_width = img_size

    # Get mean and std for reversing normalization of the images
    mean = mean.view(3, 1, 1)
    std = std.view(3, 1, 1)

    prototype_meta = model.prototype_layer.prototype_meta

    # Pre collect all prototype sample ids
    prototype_sample_ids = []
    for p in prototype_meta:
        prototype_sample_ids.append(p.sample_id)

    # Some stats
    protos_vizualized = [0] * len(prototype_meta)
    batches_not_run = 0
    batches_run = 0

    patch_info_dict = dict()

    # saving images
    img_save_path = save_loc / "prototypes"
    img_save_path.mkdir(parents=True, exist_ok=True)

    for batch_data_dict in tqdm(push_dataloader):
        search_batch_images = batch_data_dict["img"].to(device)
        # search_batch_image_paths = batch_data_dict["path"]
        sample_ids = [extract_sample_id(s) for s in batch_data_dict["sample_id"]]

        # Find any sample (sample id) in this batch matches any of the prototypes
        proto_sample_ids = []
        proto_sample_proto_indicies = []
        for proto_idx, s in enumerate(prototype_sample_ids):
            if s in sample_ids:
                proto_sample_ids.append(s)
                proto_sample_proto_indicies.append(proto_idx)

        # If there is not match in this batch, let's skip it
        if len(proto_sample_ids) == 0:
            batches_not_run += 1
            continue

        # Get model outputs
        model_outputs = model(
            (search_batch_images),
            return_prototype_layer_output_dict=True,
        )
        latent_space_size_activation_maps = model_outputs["prototype_activations"]
        if "prototype_sample_location_map" in model_outputs:
            prototype_sample_location_map = model_outputs[
                "prototype_sample_location_map"
            ]
        else:
            prototype_sample_location_map = None

        # Upsample heatmap for self activation
        pixel_space_size_activation_maps = torch.nn.Upsample(
            size=(img_height, img_width), mode="bilinear", align_corners=False
        )(latent_space_size_activation_maps)

        batches_run += 1

        # Create indicies array for position of those samples matched to a prototype (prototype samples)
        batch_indices_for_samples = [
            sample_ids.index(samp) for samp in proto_sample_ids
        ]

        proto_pixel_space_size_activation_maps = pixel_space_size_activation_maps[
            batch_indices_for_samples
        ]
        proto_latent_space_size_activation_maps = latent_space_size_activation_maps[
            batch_indices_for_samples
        ]

        # Same logic as before, plot images and save images in this loop for any protoype samples within this batch
        for idx, index_in_batch in enumerate(batch_indices_for_samples):
            sample_id = sample_ids[index_in_batch]
            proto_sample = search_batch_images[index_in_batch]
            # proto_sample_path = search_batch_image_paths[index_in_batch]
            proto_index = proto_sample_proto_indicies[idx]
            proto_info = prototype_meta[proto_index]
            if prototype_meta[proto_index].sample_id == sample_id:
                sample_hash = hash_func(proto_sample)
                if sample_hash != proto_info.hash:
                    warnings.warn(
                        f"Prototype {proto_index} hash {proto_info.hash} does not match sample {sample_hash} for {sample_id}, but they have a sample id."
                    )
            else:
                raise ValueError(f"No prototype matched the sample id {sample_id}.")

            original_img = proto_sample.detach().cpu().clone()
            original_img = (original_img * std + mean).clip(
                0, 1
            )  # reverse the normalization effect in the dataloader
            # (still need to clip for floating point error)
            original_img = original_img.numpy()
            original_img = np.transpose(original_img, [1, 2, 0])

            # Make heat map
            pixel_space_size_activation_map = proto_pixel_space_size_activation_maps[
                idx, proto_index, :, :
            ]
            pixel_space_size_activation_map = (
                pixel_space_size_activation_map
                - torch.min(pixel_space_size_activation_map)
            )
            pixel_space_size_activation_map = (
                pixel_space_size_activation_map
                / torch.max(pixel_space_size_activation_map)
            )

            (
                proto_part_lower_y,
                proto_part_upper_y,
                proto_part_right_x,
                proto_part_left_x,
            ) = find_high_activation_crop(
                pixel_space_size_activation_map.clone().detach().cpu().numpy(),
                percentile=95,
            )

            # Make boxes
            ori_img_bgr = cv2.cvtColor(np.uint8(255 * original_img), cv2.COLOR_RGB2BGR)
            ori_img_bgr_w_box = ori_img_bgr.copy()
            cv2.rectangle(
                ori_img_bgr_w_box,
                (proto_part_right_x, proto_part_lower_y),
                (proto_part_left_x - 1, proto_part_upper_y - 1),
                box_color,
                thickness=2,
            )
            ori_img_rgb_w_box = ori_img_bgr_w_box[..., ::-1]
            ori_img_rgb_w_box = np.float32(ori_img_rgb_w_box) / 255

            # TODO: change from np to only tensor use if possible here
            heatmap, overlayed_img = cv2_overlay_heatmap(
                pixel_space_size_activation_map.clone().cpu().detach().numpy(),
                original_img,
            )

            # Save images
            plt.imsave(
                img_save_path / f"proto_{proto_index}_overlayheatmap.png",
                overlayed_img,
            )
            plt.imsave(
                img_save_path / f"proto_{proto_index}_original.png",
                original_img,
            )
            plt.imsave(
                img_save_path / f"proto_{proto_index}_heatmap.png",
                heatmap,
            )
            plt.imsave(
                img_save_path / f"proto_{proto_index}_proto_bbox.png",
                ori_img_rgb_w_box,
            )

            # Record bbox information for later use
            patch_info_dict[proto_index] = [
                proto_part_lower_y,
                proto_part_upper_y,
                proto_part_right_x,
                proto_part_left_x,
                # proto_sample_path,
            ]

            protos_vizualized[proto_index] = 1
            if prototype_sample_location_map is None:
                # TODO - why would this happen?
                continue

            cur_proto_act = proto_latent_space_size_activation_maps[idx, proto_index]
            batch_argmax_cur_proto_act = list(
                custom_unravel_index(
                    torch.argmax(cur_proto_act, axis=None),
                    cur_proto_act.shape,
                )
            )
            log.info(f"batch_argmax_cur_proto_act: {batch_argmax_cur_proto_act}")
            log.info(f"cur_proto_act: {cur_proto_act}")
            best_locs_for_proto = prototype_sample_location_map[
                index_in_batch,
                :,
                :,
                batch_argmax_cur_proto_act[0],
                batch_argmax_cur_proto_act[1],
            ]
            img_with_boxes = original_img.copy()
            for part_h in range(best_locs_for_proto.shape[0]):
                for part_w in range(best_locs_for_proto.shape[1]):
                    box_coords = indices_to_upsampled_boxes(
                        best_locs_for_proto[part_h, part_w],
                        cur_proto_act.shape[-2:],
                        original_img.shape[:-1],
                    )
                    cv2.rectangle(
                        img_with_boxes,
                        box_coords[0],
                        box_coords[1],
                        (1.0, 0.0, 0.0),
                        2,
                    )

            plt.imsave(
                img_save_path / f"proto_{proto_index}_part_locs.png",
                img_with_boxes,
            )
            del heatmap, pixel_space_size_activation_map

        del pixel_space_size_activation_maps
        del latent_space_size_activation_maps

    with open(img_save_path / "patch_info_dict.json", "w") as f:
        json.dump(patch_info_dict, f)

    log.info(
        f"Batches not run because they contained no prototype samples: {batches_not_run}. Batches run: {batches_run}."
    )
    protos_vizualized = np.asarray(protos_vizualized)

    assert np.sum(protos_vizualized) == len(
        prototype_meta
    ), f"protos_vizualized is {np.sum(protos_vizualized)}; len(prototype_meta) = {len(prototype_meta)}. These should match."


def local_analysis_plotter(
    plot_save_path,
    ori_image,
    proto_actmaps_on_image,
    proto_indices,
    proto_save_dir,
    sim_scores,
    class_connections,
    num_top_protos_viewed,
    pred,
    truth,
    class_name_ref_dict,
    close_fig=True,
):
    anno_opts_cen = dict(
        xy=(0.4, 0.5), xycoords="axes fraction", va="center", ha="center"
    )
    anno_opts_symb = dict(
        xy=(1, 0.5), xycoords="axes fraction", va="center", ha="center"
    )
    anno_opts_sum = dict(xy=(0, -0.1), xycoords="axes fraction", va="center", ha="left")

    fig = plt.figure(constrained_layout=False)
    fig.set_size_inches(28, 4 * num_top_protos_viewed)

    ncols, nrows = 7, num_top_protos_viewed
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    f_axes = []
    for row in range(nrows):
        f_axes.append([])
        for col in range(ncols):
            f_axes[-1].append(fig.add_subplot(spec[row, col]))

    # FIXME: shouldn't be setting this for all of matplotlib
    plt.rcParams.update({"font.size": 14})

    for ax_num, ax in enumerate(f_axes[0]):
        if ax_num == 0:
            ax.set_title("Test image", fontdict=None, loc="left", color="k")
        elif ax_num == 1:
            ax.set_title(
                "Test image activation\nby prototype",
                fontdict=None,
                loc="left",
                color="k",
            )
        elif ax_num == 2:
            ax.set_title("Prototype", fontdict=None, loc="left", color="k")
        elif ax_num == 3:
            ax.set_title(
                "Self-activation of\nprototype", fontdict=None, loc="left", color="k"
            )
        elif ax_num == 4:
            ax.set_title("Similarity score", fontdict=None, loc="left", color="k")
        elif ax_num == 5:
            ax.set_title("Class connection", fontdict=None, loc="left", color="k")
        elif ax_num == 6:
            ax.set_title("Contribution", fontdict=None, loc="left", color="k")
        else:
            pass

    plt.rcParams.update({"font.size": 22})

    for ax in [f_axes[r][4] for r in range(nrows)]:
        ax.annotate("x", **anno_opts_symb)

    for ax in [f_axes[r][5] for r in range(nrows)]:
        ax.annotate("=", **anno_opts_symb)

    for ax in [f_axes[r][0] for r in range(nrows)]:
        ax.imshow(ori_image)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    for top_p in range(num_top_protos_viewed):
        # put info in place
        proto_idx = proto_indices[top_p]
        proto_label = torch.argmax(class_connections[:, proto_idx]).item()
        sim_score = sim_scores[top_p]
        proto_class_str = str(
            class_name_ref_dict[proto_label]
            if class_name_ref_dict is not None
            else proto_label
        )
        top_cc = class_connections[proto_label, proto_idx].item()

        for ax in [f_axes[top_p][4]]:
            ax.annotate(round(sim_score, 3), **anno_opts_cen)
            ax.set_axis_off()
        for ax in [f_axes[top_p][5]]:
            ax.annotate(str(round(top_cc, 3)) + "\n" + proto_class_str, **anno_opts_cen)
            ax.set_axis_off()
        for ax in [f_axes[top_p][6]]:
            tc = round(top_cc * sim_score, 3)
            ax.annotate("{0:.3f}".format(tc) + "\n" + proto_class_str, **anno_opts_cen)
            ax.set_axis_off()
        # put images in place
        p_img = Image.open(proto_save_dir / f"proto_{proto_idx}_original.png")
        for ax in [f_axes[top_p][2]]:
            ax.imshow(p_img)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        p_act_img = Image.open(proto_save_dir / f"proto_{proto_idx}_overlayheatmap.png")
        for ax in [f_axes[top_p][3]]:
            ax.imshow(p_act_img)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        _, act_img = cv2_overlay_heatmap(proto_actmaps_on_image[top_p], ori_image)
        for ax in [f_axes[top_p][1]]:
            ax.imshow(act_img)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
    # summary
    f_axes[2][4].annotate(
        f"This {class_name_ref_dict[int(truth)]} is classified as {class_name_ref_dict[int(pred)]}.",
        **anno_opts_sum,
    )

    plt.savefig(plot_save_path, bbox_inches="tight", pad_inches=0)
    if close_fig:
        plt.close(fig)


def local_analysis(
    model,
    save_loc,
    push_dataloader,
    eval_dataloader,
    img_size,
    mean,
    std,
    device,
    sample=None,
    num_top_protos_viewed=3,
    class_name_ref_dict=KeyReturningDict(),
    run_proto_vis=True,
):
    if run_proto_vis:
        save_prototype_images_to_file(
            model, push_dataloader, save_loc, img_size, device=device
        )

    local_ana_save_dir = save_loc / "local_analysis_saves"
    local_ana_save_dir.mkdir(parents=True, exist_ok=True)

    num_saved_analyses = 0
    mean = mean.view(3, 1, 1)
    std = std.view(3, 1, 1)

    for batch_data_dict in tqdm(eval_dataloader):
        batch_images = batch_data_dict["img"].to(device)
        # sample_ids = batch_data_dict["sample_id"]
        labels = batch_data_dict["target"]

        model_outputs = model(
            batch_images,
            return_prototype_layer_output_dict=True,
            return_similarity_score_to_each_prototype=True,
        )
        # batch size, num_protos, latent_height, latent_width
        latent_space_size_activation_maps = model_outputs["prototype_activations"]

        # batch size, num_protos
        similarity_score_to_each_prototype = model_outputs[
            "similarity_score_to_each_prototype"
        ]
        # batch size, k
        top_k_proto_sim_scores, top_k_proto_indicies = torch.topk(
            similarity_score_to_each_prototype, k=num_top_protos_viewed, dim=1
        )

        predictions = torch.argmax(model_outputs["logits"], dim=1)

        for index_in_batch in range(batch_images.shape[0]):
            ori_img = batch_images[index_in_batch].detach()
            sample_id = hash_func(ori_img)
            # ori_img is height width channel at this point
            ori_img = np.transpose(
                (ori_img.detach().cpu() * std + mean).numpy(), [1, 2, 0]
            )

            # k, latent_height, latent_width
            proto_actmaps_on_image = latent_space_size_activation_maps[
                index_in_batch, top_k_proto_indicies[index_in_batch], :, :
            ].unsqueeze(1)
            proto_actmaps_on_image_normed = (
                proto_actmaps_on_image - proto_actmaps_on_image.min()
            ) / (proto_actmaps_on_image.max() - proto_actmaps_on_image.min())
            img_height, img_width = img_size
            proto_actmaps_on_image_normed = (
                torch.nn.Upsample(
                    size=(img_height, img_width), mode="bilinear", align_corners=False
                )(proto_actmaps_on_image_normed)
                .squeeze(1)
                .clone()
                .detach()
                .cpu()
                .numpy()
            )

            local_prototype_dir = save_loc / "prototypes"
            local_prototype_dir.mkdir(parents=True, exist_ok=True)
            local_analysis_plotter(
                plot_save_path=local_ana_save_dir / f"{sample_id}.png",
                ori_image=ori_img,
                proto_actmaps_on_image=proto_actmaps_on_image_normed,
                proto_indices=top_k_proto_indicies[index_in_batch]
                .clone()
                .detach()
                .cpu()
                .numpy(),
                proto_save_dir=local_prototype_dir,
                sim_scores=top_k_proto_sim_scores[index_in_batch]
                .clone()
                .detach()
                .cpu()
                .numpy(),
                class_connections=model.prototype_prediction_head.class_connection_layer.weight,  # num_classes, num_protos
                class_name_ref_dict=class_name_ref_dict,
                num_top_protos_viewed=num_top_protos_viewed,
                pred=predictions[index_in_batch].item(),
                truth=labels[index_in_batch].item(),
            )
            del ori_img, proto_actmaps_on_image, proto_actmaps_on_image_normed

            num_saved_analyses += 1
            if sample is not None and num_saved_analyses >= sample:
                break

        if sample is not None and num_saved_analyses >= sample:
            break

        del (
            batch_images,
            labels,
            model_outputs,
            latent_space_size_activation_maps,
            similarity_score_to_each_prototype,
            top_k_proto_sim_scores,
            top_k_proto_indicies,
            predictions,
        )

    log.info(f"Completed local analysis. Saved {num_saved_analyses} analyses.")


def global_analysis(
    model,
    save_loc,
    proto_save_dir,
    proto_index,
    project_dataloader,
    img_size,
    mean,
    std,
    device,
    num_top_samples_viewed=3,
    box_color=(0, 255, 255),
):
    log.info("Running global analysis for prototype at index %s", proto_index)
    global_ana_save_dir = save_loc / "global_analysis_saves"
    global_ana_save_dir.mkdir(parents=True, exist_ok=True)
    top_k_sim_scores = []
    top_k_sample_infos = []
    all_sample_to_proto_similarity_scores = {}

    mean = mean.view(3, 1, 1)
    std = std.view(3, 1, 1)

    for batch_data_dict in tqdm(project_dataloader):
        batch_images = batch_data_dict["img"].to(device)
        labels = batch_data_dict["target"]
        sample_ids = batch_data_dict["sample_id"]

        model_outputs = model(
            # eval_normalize(batch_images),
            batch_images,
            return_prototype_layer_output_dict=True,
            return_similarity_score_to_each_prototype=True,
        )
        # batch size, num_protos, latent_height, latent_width
        latent_space_size_activation_maps = model_outputs["prototype_activations"]

        img_height, img_width = img_size
        pixel_space_size_activation_maps = torch.nn.Upsample(
            size=(img_height, img_width), mode="bilinear", align_corners=False
        )(latent_space_size_activation_maps)

        # batch size, num_protos
        similarity_score_to_each_prototype = model_outputs[
            "similarity_score_to_each_prototype"
        ]
        log.debug("all similarity scores %s", similarity_score_to_each_prototype.shape)
        sample_to_proto_similarity_scores = similarity_score_to_each_prototype[
            :, proto_index
        ]
        log.debug("prototype similarity scores %s", sample_to_proto_similarity_scores)

        for idx, sample_id in enumerate(sample_ids):
            all_sample_to_proto_similarity_scores[extract_sample_id(sample_id)] = (
                sample_to_proto_similarity_scores.detach().cpu().numpy()[idx]
            )

        for in_batch_idx in range(batch_images.shape[0]):
            pixel_space_size_activation_map = pixel_space_size_activation_maps[
                in_batch_idx, proto_index, :, :
            ]
            pixel_space_size_activation_map = (
                pixel_space_size_activation_map
                - torch.min(pixel_space_size_activation_map)
            )
            pixel_space_size_activation_map = (
                pixel_space_size_activation_map
                / torch.max(pixel_space_size_activation_map)
            )

            # reverse normalization in dataloader
            original_img = np.transpose(
                (
                    batch_images[in_batch_idx].clone().cpu().detach() * std + mean
                ).numpy(),
                [1, 2, 0],
            )
            heatmap, overlayed_img = cv2_overlay_heatmap(
                pixel_space_size_activation_map.clone().cpu().detach().numpy(),
                original_img,
            )

            (
                proto_part_lower_y,
                proto_part_upper_y,
                proto_part_right_x,
                proto_part_left_x,
            ) = find_high_activation_crop(
                pixel_space_size_activation_map.clone().detach().cpu().numpy(),
                percentile=95,
            )

            packet = [
                (
                    proto_part_lower_y,
                    proto_part_upper_y,
                    proto_part_right_x,
                    proto_part_left_x,
                ),
                original_img,
                overlayed_img,
                labels[in_batch_idx].item(),
                sample_ids[in_batch_idx],
                sample_to_proto_similarity_scores[in_batch_idx].item(),
            ]

            if len(top_k_sim_scores) < num_top_samples_viewed:
                top_k_sim_scores.append(
                    sample_to_proto_similarity_scores[in_batch_idx].item()
                )
                top_k_sample_infos.append(packet)
            else:
                min_value = min(top_k_sim_scores)
                min_index = top_k_sim_scores.index(min_value)

                if sample_to_proto_similarity_scores[in_batch_idx] > min_value:
                    top_k_sim_scores[min_index] = sample_to_proto_similarity_scores[
                        in_batch_idx
                    ]
                    top_k_sample_infos[min_index] = packet

            combined = list(zip(top_k_sim_scores, top_k_sample_infos))
            combined_sorted = sorted(combined, key=lambda x: x[0])
            top_k_sim_scores = [element[0] for element in combined_sorted]
            top_k_sample_infos = [element[1] for element in combined_sorted]

        del (
            model_outputs,
            pixel_space_size_activation_map,
            latent_space_size_activation_maps,
        )

    sorted_items = sorted(
        all_sample_to_proto_similarity_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    _ = sorted_items[:num_top_samples_viewed]  # was topk

    actual_proto_sample_id = extract_sample_id(
        model.prototype_layer.prototype_meta[proto_index].sample_id
    )
    log.debug(
        "%s %s, %s",
        proto_index,
        actual_proto_sample_id,
        all_sample_to_proto_similarity_scores[actual_proto_sample_id],
    )
    log.debug(top_k_sample_infos)
    p_img = Image.open(proto_save_dir / f"proto_{proto_index}_proto_bbox.png")

    selected_samples_to_target_proto = [p_img]
    selected_samples_to_target_proto_heatmap = [p_img]
    # titles = [f"Prototype {proto_index}"]
    titles = ["Prototype"]

    for idx, packet in reversed(list(enumerate(top_k_sample_infos))):
        bbox, original_img, overlayed_img, label, sample_id, sim_score = packet
        (
            proto_part_lower_y,
            proto_part_upper_y,
            proto_part_right_x,
            proto_part_left_x,
        ) = bbox

        ori_img_bgr = cv2.cvtColor(np.uint8(255 * original_img), cv2.COLOR_RGB2BGR)
        ori_img_bgr_w_box = ori_img_bgr.copy()
        cv2.rectangle(
            ori_img_bgr_w_box,
            (proto_part_right_x, proto_part_lower_y),
            (proto_part_left_x - 1, proto_part_upper_y - 1),
            box_color,
            thickness=2,
        )
        ori_img_rgb_w_box = ori_img_bgr_w_box[..., ::-1]
        ori_img_rgb_w_box = np.float32(ori_img_rgb_w_box) / 255
        selected_samples_to_target_proto.append(ori_img_rgb_w_box)
        # titles.append(f"Sample {num_top_samples_viewed - idx} \nSimilarity score {round(sim_score, 3)}")
        titles.append(f"Sample {num_top_samples_viewed - idx}")

        selected_samples_to_target_proto_heatmap.append(overlayed_img)

    N = len(selected_samples_to_target_proto)

    _ = plt.figure(figsize=(5 * N + 2, 5))
    gs = gridspec.GridSpec(1, N + 1, width_ratios=[1, 0.3] + [1] * (N - 1))

    axes = []
    for i in range(N):
        if i == 1:
            axes.append(plt.subplot(gs[i + 1]))
        else:
            axes.append(plt.subplot(gs[i if i == 0 else i + 1]))

    for i, ax in enumerate(axes):
        ax.imshow(selected_samples_to_target_proto[i])
        # ax.set_title(titles[i], fontsize=20)
        ax.axis("off")

    plt.savefig(
        global_ana_save_dir / f"{proto_index}_view_{num_top_samples_viewed}.png",
        bbox_inches="tight",
    )
    plt.clf()

    _ = plt.figure(figsize=(5 * N + 2, 5))
    gs = gridspec.GridSpec(1, N + 1, width_ratios=[1, 0.3] + [1] * (N - 1))

    axes = []
    for i in range(N):
        if i == 1:
            axes.append(plt.subplot(gs[i + 1]))
        else:
            axes.append(plt.subplot(gs[i if i == 0 else i + 1]))

    for i, ax in enumerate(axes):
        ax.imshow(selected_samples_to_target_proto_heatmap[i])
        # ax.set_title(titles[i], fontsize=20)
        ax.axis("off")

    plt.savefig(
        global_ana_save_dir
        / f"{proto_index}_view_{num_top_samples_viewed}_heatmap.png",
        bbox_inches="tight",
    )
    plt.clf()


def recover_proto_meta(model, push_dataloader, model_save_loc, device):
    prototype_meta = model.prototype_layer.prototype_meta

    if len(prototype_meta) == 0:
        prototype_meta = {}
        prototype_tensors = model.prototype_layer.prototype_tensors
        prototype_matches = [0] * model.prototype_layer.num_prototypes
        for batch_data_dict in tqdm(push_dataloader):
            search_batch_images = batch_data_dict["img"].to(device)
            sample_ids = batch_data_dict["sample_id"]

            search_batch_latent_vectors = model.add_on_layers(
                model.backbone(search_batch_images)
            )
            log.info(search_batch_latent_vectors.shape)
            log.info(prototype_tensors.shape)

            for in_batch_idx in range(search_batch_latent_vectors.shape[0]):
                difference = (
                    search_batch_latent_vectors[in_batch_idx] - prototype_tensors
                )
                zero_vector = torch.zeros(256)
                is_zero_vector = torch.isclose(
                    difference, zero_vector.view(1, -1, 1, 1)
                )
                is_zero_vector_reduced = is_zero_vector.all(dim=1, keepdim=True)

                for match_idx in torch.nonzero(is_zero_vector_reduced):
                    prototype_matches[match_idx] += 1
                    prototype_meta[sample_ids[in_batch_idx]] = hash_func(
                        search_batch_images[in_batch_idx]
                    )

        log.info(sum(prototype_matches) / len(prototype_matches))
        log.info(len(prototype_meta))

        model.prototype_layer.prototype_meta = prototype_meta

        torch.save(
            obj=model,
            f=model_save_loc,
        )


def reproject_prototypes(model, output_path, project_dataloader):
    """
    Project the prototypes and save the reprojected model to the output path.
    """

    # Combine parent directory with the new file name
    reprojected_model_save_loc = output_path / "reprojected_model.pth"
    reprojected_model_infodict_save_loc = output_path / "reprojected_proto_info_dict.pt"
    reprojected_model_save_loc.parent.mkdir(parents=True, exist_ok=True)

    log.info("Reprojecting model and saving to %s", reprojected_model_save_loc)

    if reprojected_model_infodict_save_loc.exists():
        # Allow for caching of the prototype info dict between runs
        prototype_meta = torch.load(reprojected_model_infodict_save_loc)

        if len(prototype_meta) > 0:
            model.prototype_layer.prototype_meta = prototype_meta
            log.info(
                f"Saved exist, reloading info dict len={len(model.prototype_layer.prototype_meta)}"
            )
            return model

    # push old model to get proto hashes (kind of hacky)
    # FIXME: check that this didn't change results somehow
    model.project(project_dataloader)
    # log.info("description of protos without vis: ", model.describe_prototypes())
    log.info("Project complete, saving model to %s", reprojected_model_save_loc)

    torch.save(
        obj=model,
        f=reprojected_model_save_loc,
    )
    torch.save(
        obj=model.prototype_layer.prototype_meta,
        f=reprojected_model_infodict_save_loc,
    )
    log.debug(
        f"After project info dict len={len(model.prototype_layer.prototype_meta)}"
    )
    log.info(
        "Reloading saved model after project from %s", str(reprojected_model_save_loc)
    )

    model = torch.load(reprojected_model_save_loc)
    log.info(
        f"After reloading project info dict len={len(model.prototype_layer.prototype_meta)}"
    )

    return model


def sample_activations(folder_path, n, title):
    """
    Example usage
    sample_activations(
        'protopnext/all_analysis_2/vis_c3tskyq5_90_project_0.7196_reprojected.pth/global_analysis_saves/',
        15,
        ' Prototype' + ' '*79+'Nearest 10 Samples'
    )
    """

    # List all files in the directory
    files = pathlib.Path(folder_path)

    # Filter out files that do not end with ".png" or end with "_heatmap.png"
    images = [
        file
        for file in files
        if file.endswith(".png") and not file.endswith("_heatmap.png")
    ]

    # Randomly select N images
    selected_images = random.sample(images, n)

    # Create figure and axes
    _, axs = plt.subplots(n, 1, figsize=(20, n * 2))

    # Check if we have multiple axes or just one
    if n == 1:
        axs = [axs]  # Make it iterable if there is only one subplot

    # Plot each image
    for idx, (ax, image_name) in tqdm(enumerate(zip(axs, selected_images)), total=n):
        img_path = folder_path / image_name
        img = Image.open(img_path)
        ax.imshow(img)
        if idx == 0:
            ax.set_title(title, loc="left", fontsize=21)
        ax.axis("off")  # Turn off axis

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.01)  # Adjust the horizontal space between images
    plt.savefig(f"plops/global_choose_{n}.png", dpi=200, bbox_inches="tight")


@click.command(
    "viz",
    help="Visualize the prototypes of a model",
)
# TODO - each analysis should be it's own command in a command group with its own options
# but this initialization needs to be refactored to support that
@click.argument(
    "analysis_type",
    type=click.Choice(["global", "local", "render-prototypes"], case_sensitive=False),
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path("analyses/"),
    help="Directory for output files",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=True,
    help="Path to the model",
)
@click.option(
    "--dataset",
    type=click.Choice(
        ["cub200", "cub200-cropped", "cars", "dogs", "cifar10", "mammo", "shapes_2d"],
        case_sensitive=False,
    ),
    required=True,
    help="Dataset to use",
)
@click.option("--sample", type=int, default=None, help="Sample size to use")
def run(
    *,
    model_path: pathlib.Path,
    dataset: str,
    analysis_type: str,
    output_dir: pathlib.Path = pathlib.Path("analyses/"),
    batch_size: int = 4,
    global_analysis_top_samples: int = 10,
    sample: int = None,
    local_analysis_top_comparisons: int = 3,
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
):
    """
    Run Global and Local Analyses for a trained model.

    This should be broken into individual functions for each type of analysis in the future.
    """
    log.info("load model from %s", str(model_path))
    model = torch.load(model_path, map_location=device)
    model.prune_duplicate_prototypes()
    model.eval()
    with torch.no_grad():

        # TODO - why is this necessary?
        model.prototype_layer.latent_spatial_size = None

        train_dataloaders = datasets.training_dataloaders(
            dataset,
            batch_sizes={"train": batch_size, "val": batch_size, "project": batch_size},
        )

        val_dataloader = train_dataloaders.val_loader
        project_dataloader = train_dataloaders.project_loader

        if len(model.prototype_layer.prototype_meta) == 0:
            model = reproject_prototypes(model, output_dir, project_dataloader)

        vis_save_loc = (
            output_dir / f"vis_{model_path.parent.name + '_' + model_path.name}"
        )
        vis_save_loc.mkdir(parents=True, exist_ok=True)

        class_name_ref_dict = KeyReturningDict()

        # FIXME: degrading this mapping until we have a general solution
        # file_path = ""
        # # Open and read the file
        # with open(file_path, "r") as file:
        #     for index, line in enumerate(
        #         file, start=1
        #     ):  # start=1 starts the index from 1
        #         parts = line.strip().split(
        #             "."
        #         )  # Split by dot to separate the number and bird name
        #         if len(parts) > 1:
        #             class_name_ref_dict[index] = parts[
        #                 1
        #             ]  # After the dot is the bird name
        # Display the dictionary
        # log.info(class_name_ref_dict)

        if analysis_type == "render-prototypes":
            log.info(f"rendering prototypes to {vis_save_loc}")

            save_prototype_images_to_file(
                model,
                project_dataloader,
                vis_save_loc,
                train_dataloaders.image_size,
                train_dataloaders.train_distribution.mean,
                train_dataloaders.train_distribution.std,
                device=device,
            )

            log.info("Completed rendering of prototypes.")

        elif analysis_type == "local":
            local_analysis(
                model,
                save_loc=vis_save_loc,
                class_name_ref_dict=class_name_ref_dict,
                push_dataloader=project_dataloader,
                eval_dataloader=val_dataloader,
                img_size=train_dataloaders.image_size,
                mean=train_dataloaders.train_distribution.mean,
                std=train_dataloaders.train_distribution.std,
                device=device,
                sample=sample,
                num_top_protos_viewed=local_analysis_top_comparisons,
                run_proto_vis=False,
            )

        elif analysis_type == "global":
            for proto_index in range(model.prototype_layer.num_prototypes):
                if sample is not None and proto_index >= sample:
                    break

                global_analysis(
                    model,
                    save_loc=vis_save_loc,
                    proto_save_dir=vis_save_loc / "prototypes",
                    proto_index=proto_index,
                    project_dataloader=project_dataloader,
                    img_size=train_dataloaders.image_size,
                    mean=train_dataloaders.train_distribution.mean,
                    std=train_dataloaders.train_distribution.std,
                    device=device,
                    num_top_samples_viewed=global_analysis_top_samples,
                )

            log.info("Completed global analysis.")
