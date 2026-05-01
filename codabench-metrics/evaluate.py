#!/usr/bin/env python
import sys
import os
import json
from glob import glob
import time
import datetime
import math

import numpy as np
import pyproj
from pyproj.database import query_utm_crs_info
from pyproj.aoi import AreaOfInterest
from scipy.spatial.transform import Rotation

sglob = lambda p: sorted(glob(p))

print(f"CURRENT TIME: {datetime.datetime.now().time()}")


def geodetic_to_enu(lat, lon, alt, lat_org, lon_org, alt_org):
    """
    convert LLA to ENU
    :params lat, lon, alt: input LLA coordinates
    :params lat_org, lon_org, alt_org: LLA of the origin of the local ENU coordinate system
    :return: east, north, up coordinate
    """
    transformer = pyproj.Transformer.from_crs(
        {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        always_xy=True,
    )
    x, y, z = transformer.transform(lon, lat, alt, radians=False)
    x_org, y_org, z_org = transformer.transform(
        lon_org, lat_org, alt_org, radians=False
    )
    vec = np.array([[x - x_org, y - y_org, z - z_org]]).T

    rot1 = Rotation.from_euler(
        "x", -(90 - lat_org), degrees=True
    ).as_matrix()  # angle*-1 : left handed *-1
    rot3 = Rotation.from_euler(
        "z", -(90 + lon_org), degrees=True
    ).as_matrix()  # angle*-1 : left handed *-1

    rot_matrix = rot1.dot(rot3)
    enu = rot_matrix.dot(vec).T.ravel()
    return enu.T


def utm_epsg_from_wgs84(latitude, longitude):
    """
    get UTM EPSG code for input WGS84 coordinate
    :param latitude, longitude: WGS84 coordinate in degrees
    :return: UTM EPSG code
    """
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=longitude,
            south_lat_degree=latitude,
            east_lon_degree=longitude,
            north_lat_degree=latitude,
        ),
    )
    utm_epsg = int(utm_crs_list[0].code)
    return utm_epsg



def lla_to_utm(lat, lon, alt, utm_epsg=None):
    """
    convert WGS84 coordinate to UTM
    :param lat,lon,alt: WGS84 coordinate
    :param utm_epsg: pre-calculated UTM EPSG
    :return: UTM x,y,z
    """
    if utm_epsg is None:
        utm_epsg = utm_epsg_from_wgs84(lat, lon)
    transformer = pyproj.Transformer.from_crs(4326, utm_epsg, always_xy=True)
    x, y, z = transformer.transform(lon, lat, alt)
    return x, y, z


def opk_to_rotation(opk_degrees):
    """
    calculate a rotation matrix given euler angles
    :params opk: list of [omega, phi, kappa] angles (degrees)
    :return: rotation matrix
    """
    opk = np.radians(opk_degrees)
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(opk[0]), -math.sin(opk[0])],
            [0, math.sin(opk[0]), math.cos(opk[0])],
        ]
    )
    R_y = np.array(
        [
            [math.cos(opk[1]), 0, math.sin(opk[1])],
            [0, 1, 0],
            [-math.sin(opk[1]), 0, math.cos(opk[1])],
        ]
    )
    R_z = np.array(
        [
            [math.cos(opk[2]), -math.sin(opk[2]), 0],
            [math.sin(opk[2]), math.cos(opk[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_x, np.dot(R_y, R_z))
    return R


def opk_to_ypr(lat, lon, alt, o, p, k):
    """
    convert omega, phi, and kappa angles to yaw, pitch, and roll angles
    :param lat, lon, alt: nad83 coordinate used to determine axis normals
    :param o, p, k: omega, phi, kappa  angles in degrees
    :return: yaw, pitch, and roll angles in degrees
    """

    opk = [o, p, k]
    ceb = opk_to_rotation(opk)
    cbb = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    # compute x, y, and z normals; x is North in UTM grid
    delta = 1e-6
    p1 = np.array(lla_to_utm(lat + delta, lon, alt))
    p2 = np.array(lla_to_utm(lat - delta, lon, alt))
    xnp = p1 - p2
    m = np.linalg.norm(xnp)
    xnp /= m
    znp = np.array([0, 0, -1])
    ynp = np.cross(znp, xnp)
    # convert to OPK
    cen = np.array([xnp, ynp, znp]).T
    cnb = np.linalg.inv(cen).dot(ceb).dot(np.linalg.inv(cbb))
    R = cnb
    # additional Ry rotation -90
    Ry = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    R = R.dot(Ry)
    r = Rotation.from_matrix(R)
    [yaw, pitch, roll] = r.as_euler("ZYX", degrees=True)
    return yaw, pitch, roll




def load_metadata(path):
    """
    Loads metadata jsons at path

    :param path: The path that contains metadata jsons
    :return:
    """
    all_json_paths = sglob(os.path.join(path, '*.json'))
    all_metadata = {}
    for json_path in all_json_paths:
        with open(json_path, 'r') as f:
            all_metadata[os.path.basename(json_path)] = json.load(f)
    return all_metadata


if __name__ == "__main__":
    # define input and output paths
    start = time.time()
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    submission_path = os.path.join(input_path, 'res')
    reference_path = os.path.join(input_path, 'ref')

    reference_datasets = sorted(os.listdir(reference_path))
    summary_results = {}

    for dataset in reference_datasets:
        print(f"Evaluating dataset {dataset}")
        dataset_reference_path = os.path.join(reference_path, dataset)
        # turns, for example, WRIVA-CVGL-DEV-001 -> dev-001-geo-rmse-se90 for leaderboard key
        dataset_leaderboard_geo_name = "-".join(dataset.split("-")[2:4]).lower() + "-geo-rmse-se90"
        dataset_leaderboard_heading_name = "-".join(dataset.split("-")[2:4]).lower() + "-heading-rmse-se90"

        reference_metadata = load_metadata(os.path.join(dataset_reference_path, "reference"))
        submission_metadata = load_metadata(os.path.join(submission_path, dataset))
        ks = list(reference_metadata.keys())
        ref_coords = np.array([
            [reference_metadata[k]["extrinsics"]["lat"],
             reference_metadata[k]["extrinsics"]["lon"]]
            for k in ks
        ])

        sub_coords = np.array([
            [submission_metadata[k]["lat"],
             submission_metadata[k]["lon"]]
            for k in ks
        ])

        assert len(ref_coords) == len(sub_coords)
        lat_origin, lon_origin = ref_coords[0]

        ref_coords_enu = np.array([
            geodetic_to_enu(lat, lon, 0.0, lat_origin, lon_origin, 0.0)
            for lat, lon in ref_coords
        ])
        sub_coords_enu = np.array([
            geodetic_to_enu(lat, lon, 0.0, lat_origin, lon_origin, 0.0)
            for lat, lon in sub_coords
        ])
        rmses = np.sqrt(((ref_coords_enu - sub_coords_enu) ** 2).sum(axis=1))
        summary_results[dataset_leaderboard_geo_name] = np.percentile(rmses, 90)

        if all("heading" in sub_keys and "pitch" in sub_keys for sub_keys in submission_metadata.values()):
            ref_opk = np.array([
                [reference_metadata[k]["extrinsics"]["omega"],
                 reference_metadata[k]["extrinsics"]["phi"],
                 reference_metadata[k]["extrinsics"]["kappa"]]
                for k in ks
            ])
            ref_ypr = np.array([
                # For evaluation, we dont care about roll
                [yaw, pitch, 0.0]
                for (lat, lon), (omega, phi, kappa) in zip(ref_coords, ref_opk)
                for yaw, pitch, _ in [opk_to_ypr(lat, lon, 0.0, omega, phi, kappa)]
            ])
            sub_ypr = np.array([
                [submission_metadata[k]["heading"],
                 submission_metadata[k]["pitch"],
                 # 0.0 is for the roll
                 0.0]
                for k in ks
            ])
            heading_rmses = np.sqrt(((ref_ypr - sub_ypr) ** 2).sum(axis=1))
            summary_results[dataset_leaderboard_heading_name] = np.percentile(heading_rmses, 90)
        else:
            summary_results[dataset_leaderboard_geo_name] = 0.0

    print('Writing scores...')
    fid = open(os.path.join(output_path, 'scores.txt'), "w")
    for key, score in summary_results.items():
        fid.write(f"{key}: {score}\n")
    fid.close()

    end = time.time()
    print(f'Finished. Time to complete: {end - start} seconds.')
