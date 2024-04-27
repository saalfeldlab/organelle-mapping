from fly_organelles.model import StandardUnet
from fly_organelles.train import make_train_pipeline
import gunpowder as gp
import logging
import yaml
import click

# logging.basicConfig(level=logging.DEBUG)
# loggp = logging.getLogger("gunpowder.nodes.pad")
# loggp.setLevel(logging.DEBUG)


def run(iterations, labels, label_weights, datasets):
    model = StandardUnet(len(labels))
    
    voxel_size = (8, 8, 8)
    input_size = gp.Coordinate((178, 178, 178)) * gp.Coordinate(voxel_size)
    output_size = gp.Coordinate((56, 56, 56)) * gp.Coordinate(voxel_size)
    buffer = gp.Coordinate((24*6,24*6,24*6))
    pad_width_out = output_size/2.
    pad_width_in = (input_size - output_size)/2. + pad_width_out + buffer
     
    pipeline = make_train_pipeline(
        model,
        labels=labels,
        label_weights=label_weights,
        datasets=datasets,
        pad_width_in=pad_width_in,
        pad_width_out=pad_width_out,
        sampling=voxel_size,
    )
    

    with gp.build(pipeline) as pp:
        for i in range(iterations):
            request = gp.BatchRequest()
            request.add(gp.ArrayKey("OUTPUT"), output_size, voxel_size=gp.Coordinate(voxel_size))
            request.add(gp.ArrayKey("RAW"), input_size, voxel_size= gp.Coordinate(voxel_size))
            request.add(gp.ArrayKey("LABELS"), output_size, voxel_size=gp.Coordinate(voxel_size))
            request.add(gp.ArrayKey("MASK"), output_size, voxel_size = gp.Coordinate(voxel_size))
            pp.request_batch(request)

@click.command()
@click.argument("data-config", type=click.File("rb"))
@click.argument("iterations", type=int)
def main(data_config):
    labels = ["organelle", "all_mem"]
    label_weights = [
        1.0 / len(labels),
    ] * len(labels)
    #with open(yaml_file, "r") as data_yaml:
    datasets = yaml.safe_load(data_config)
        #label_stores, raw_stores, crop_copies = read_data_yaml(data_yaml)
    run(iterations, labels, label_weights, datasets)
