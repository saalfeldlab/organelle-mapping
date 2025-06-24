import logging

import click
import gunpowder as gp
import numpy as np
import yaml

from fly_organelles.model import StandardUnet
from fly_organelles.train import make_train_pipeline, make_data_pipeline

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
# loggp = logging.getLogger("gunpowder.nodes.pad")
# loggp.setLevel(logging.DEBUG)
heads_key = ["final_conv.bias", "final_conv.weight"]

def set_weights(model, weights, old_head, new_head, else_map={}):
    logger.warning(
        f"loading weights old_head {old_head}, new_head: {new_head}"
    )
    for key in weights.keys():
        if key in heads_key:
            logger.warning(f"key: {key}")
            weights[key] = match_heads(weights[key], model.state_dict()[key], old_head, new_head, else_map)
    return weights

def match_heads(checkpoint_weights, current_model_weights, old_head, new_head, else_map={}):
    for new_index,label in enumerate(new_head):
        if label in old_head:
            old_index = old_head.index(label)
            new_value = checkpoint_weights[old_index]
            current_model_weights[new_index] = new_value
            logger.warning(f"matched head for {label}.")
        elif label in else_map.keys():
            old_index = old_head.index(else_map[label])
            new_index = new_head.index(label)
            new_value = checkpoint_weights[old_index]
            current_model_weights[new_index] = new_value
            logger.warning(f"matched head for {label} with {else_map[label]}.")
    
    return current_model_weights




def run(model,iterations, labels, label_weights, datasets,voxel_size = (8, 8, 8),batch_size = 14, l_rate=0.5e-4, log_dir = "logs",  affinities = False, affinities_map = None, min_mask = None, input_size = gp.Coordinate((178, 178, 178)), output_size = gp.Coordinate((56, 56, 56)), distance_sigma=None ):
    input_size = gp.Coordinate(input_size) * gp.Coordinate(voxel_size)
    output_size = gp.Coordinate(output_size) * gp.Coordinate(voxel_size)
    displacement_sigma = gp.Coordinate((24, 24, 24))
    # max_in_request = gp.Coordinate((np.ceil(np.sqrt(sum(input_size**2))),)*len(input_size)) + displacement_sigma * 6
    max_out_request = (
        gp.Coordinate((np.ceil(np.sqrt(sum(output_size**2))),) * len(output_size)) + displacement_sigma * 6
    )
    pad_width_out = output_size / 2.0

    pipeline = make_train_pipeline(
        model,
        labels=labels,
        label_weights=label_weights,
        datasets=datasets,
        pad_width_out=pad_width_out,
        sampling=voxel_size,
        max_out_request=max_out_request,
        displacement_sigma=displacement_sigma,
        input_size=input_size,
        output_size=output_size,
        batch_size=batch_size,
        l_rate= l_rate,
        log_dir=log_dir, 
        affinities = affinities,
        affinities_map = affinities_map,
        min_mask = min_mask,
        distance_sigma = distance_sigma,
    )

    request = gp.BatchRequest()
    request.add(gp.ArrayKey("OUTPUT"), output_size, voxel_size=gp.Coordinate(voxel_size))
    request.add(gp.ArrayKey("RAW"), input_size, voxel_size=gp.Coordinate(voxel_size))
    request.add(gp.ArrayKey("LABELS"), output_size, voxel_size=gp.Coordinate(voxel_size))
    request.add(gp.ArrayKey("MASK"), output_size, voxel_size=gp.Coordinate(voxel_size))
    with gp.build(pipeline) as pp:
        for i in range(iterations):
            pp.request_batch(request)


@click.command()
@click.argument("data-config", type=click.File("rb"))
@click.argument("iterations", type=int)
@click.option(
    "--labels", "-l", multiple=True, type=str, help="List of labels to train for.", default=["organelle", "all_mem"]
)
@click.option("--label_weights", "-lw", multiple=True, type=float)
def main(data_config, iterations, labels, label_weights=None):
    if not label_weights:
        label_weights = [
            1.0 / len(labels),
        ] * len(labels)
    else:
        if len(label_weights) != len(labels):
            msg = (
                f"If label weights are specified ({type(label_weights)}),"
                f"they need to be of the same length as the list of labels ({len(labels)})"
            )
            raise ValueError(msg)
        normalizer = np.sum(label_weights)
        label_weights = [lw / normalizer for lw in label_weights]
    logger.info(
        f"Running training for the following labels:"
        f"{', '.join([f'{lbl} ({lblw:.4f})' for lbl,lblw in zip(labels,label_weights)])}"
    )
    # with open(yaml_file, "r") as data_yaml:
    datasets = yaml.safe_load(data_config)
    # label_stores, raw_stores, crop_copies = read_data_yaml(data_yaml)
    run(iterations, labels, label_weights, datasets)
