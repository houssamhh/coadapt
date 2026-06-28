import os
import interpret_pointcloud as pc_interpreter
import glob
import random
import re
import csv
import logging


# ---------Logging------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("dataset_generation.log"),
        logging.StreamHandler()
    ]
)

pattern = re.compile(r"^\d+\.pcd$")             ## pattern to match against for pcd files

def get_all_pcd_files(root_dir, pattern):
    pcd_files = []
    for path in glob.glob(os.path.join(root_dir, "**", "*.pcd"), recursive=True):
        if pattern.search(os.path.basename(path)):
            pcd_files.append(path)
    return pcd_files


def get_obstacles_description(pcd_path):
    pcd = pc_interpreter.load_point_cloud(pcd_path)
    try:
        ground, objects = pc_interpreter.segment_ground(pcd)
    except TypeError as e:
        print(f"Couldn't segment objects for file {pcd_path}")
        return "No description", "No description"
    clusters = pc_interpreter.cluster_obstacles(objects)

    obstacles_description = pc_interpreter.describe_obstacles(clusters)             # returned as a dict
    textual_obstacles_description = pc_interpreter.obstacles_to_text(obstacles_description)     # return for the LLM to understand
    return obstacles_description, textual_obstacles_description

def create_csv_file(root_dir, output_file):
    header = ["file_path", "textual_description", "distance", "obstacles_within_d_meters"]
    with open(output_file, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(header)
    pcd_files = get_all_pcd_files(root_dir, pattern)
    for file in pcd_files:
        try:
            obstacles_description, textual_obstacles_description = get_obstacles_description(file)
            distance = random.randint(2, 15)
            obstacles_within_d_meters = pc_interpreter.get_obstacles_within_d_meters(obstacles_description, distance)
            row = [file, textual_obstacles_description, distance, obstacles_within_d_meters]
        except Exception as e:
            logging.error(f"[ERROR] Failure on file: {file}\nReason: {str(e)}")
            row = [file, "Error processing file", -1, []]
        with open(output_file, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(row)


if __name__ == "__main__":
    dirs = {"train": r"../V2X-R/decompress/train", "test": r"../V2X-R/decompress/test", "validate": r"../V2X-R/decompress/validate"}

    for type, dir in dirs.items():
        logging.info(f"Creating dataset for {type} data...")
        create_csv_file(dir, f"../V2X-R/{type}.csv")
        logging.info("Done.")
    logging.info("Created all datasets.")
