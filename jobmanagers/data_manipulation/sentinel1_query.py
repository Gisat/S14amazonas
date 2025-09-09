from tondortools.creodias import AVAILABLE_STATUSES, CatalogStatus, query_catalog
from osgeo import ogr
from pathlib import Path
import subprocess
import re
from datetime import datetime
import json


def find_latest_sceneinfo(scene_info_list):
    latest_scene_info = None
    latest_time = None
    for scene_info_list_item in scene_info_list:
        scene_path = scene_info_list_item["path"]
        scene_manifest_path = Path(scene_info_list_item["path"]).joinpath('manifest.safe')
        if scene_manifest_path.exists():
            args = ["grep", "GRD Post Processing", str(scene_manifest_path)]
            output = subprocess.check_output(args)
            # define the regular expression pattern to match the timestamp
            pattern = r'stop="([0-9-]+T[0-9:.]+)"'
            # define the input string
            # search for the timestamp in the input string using the regular expression
            match = re.search(pattern, output.decode())
            # extract the matched timestamp string
            timestamp_str = match.group(1)
            # print the extracted timestamp string
            current_time = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f')
            print(f"processing time for {scene_info_list_item['title']}: {current_time}")
            if latest_time == None:
                latest_time = current_time
                latest_scene_info = scene_info_list_item
                continue
            if current_time > latest_time:
                latest_time = current_time
                latest_scene_info = scene_info_list_item
    print(f"selecting {latest_scene_info['title']}")
    return latest_scene_info



def query_sentinel1(xmin, ymin, xmax, ymax, start_date, final_date):
    # Query the catalog.
    query_args = {"productType": "GRD",
                  "sensorMode": "IW",
                  "box": "{},{},{},{}".format(xmin, ymin, xmax, ymax),
                  "startDate": start_date.isoformat(),
                  "completionDate": final_date.isoformat(),
                  "status": "all"}
    features = query_catalog("Sentinel1", query_args)

    # Prepare scene infos.
    scene_infos_safe = []
    scene_infos_cogs = []
    for feature in features:

        if feature["properties"]["status"].isnumeric():
            status = CatalogStatus(int(feature["properties"]["status"]))
        elif isinstance(feature["properties"]["status"], str):
            status = feature["properties"]["status"]
        else:
            raise Exception(f"unknown status: {status}")

        if status not in AVAILABLE_STATUSES:
            print("Ignoring scene {:s}, while it has unavailable status {:s}."
                        .format(feature["properties"]["title"], repr(status)))
        else:
            # Prepare and check scene geometry.
            scene_geom = ogr.CreateGeometryFromJson(json.dumps(feature["geometry"]))
            scene_geom_type = scene_geom.GetGeometryType()
            if scene_geom.GetGeometryCount() > 1:
                print("Ignoring scene {:s}, while it has {:d} geometries."
                            .format(feature["properties"]["title"], scene_geom.GetGeometryCount()))
            elif scene_geom_type not in (ogr.wkbPolygon, ogr.wkbMultiPolygon):
                print("Ignoring scene {:s}, while it has unexpected geometry type {:d}."
                            .format(feature["properties"]["title"], scene_geom_type))
            else:
                if 'CARD_BS' in feature["properties"]["title"]:
                    continue

                if not Path(feature["properties"]["productIdentifier"]).exists():
                    continue

                # Write the feature into the features
                print(f"accepting scene: {feature['properties']['title']} with status: {feature['properties']['status']}")

                scene_id = feature["properties"]["title"].split('_')
                scene_id = '_'.join(scene_id[0:8])

                scene_info = {"title": feature["properties"]["title"],
                              "scene_id": scene_id,
                              "startDate": feature["properties"]["startDate"][:19],
                              "relativeOrbitNumber": feature["properties"]["relativeOrbitNumber"],
                              "OrbitDirec": str(feature["properties"]["orbitDirection"]).lower(),
                              "path": Path(feature["properties"]["productIdentifier"]),
                              "geom": scene_geom}

                if 'COG' in feature["properties"]["title"]:
                    scene_infos_cogs.append(scene_info)
                else:
                    scene_infos_safe.append(scene_info)


    # from safe list, find the latest date for those with same datatime and diff hex code
    scene_infos_safe_grouped = {}
    for scene_info in scene_infos_safe:
        if scene_info["scene_id"] in scene_infos_safe_grouped.keys():
            scene_infos_safe_grouped[scene_info["scene_id"]] += [scene_info]
        else:
            scene_infos_safe_grouped[scene_info["scene_id"]] = [scene_info]

    scene_infos_safe_condensed = []
    for scene_id, scene_info_list in scene_infos_safe_grouped.items():
        if len(scene_info_list) == 1:
            scene_infos_safe_condensed.append(scene_info_list[0])
        else:
            latest_scene_info = find_latest_sceneinfo(scene_info_list)
            scene_infos_safe_condensed.append(latest_scene_info)

    # from cogsafe list, find the latest date for those with same datatime and diff hex code
    scene_infos_cogsafe_grouped = {}
    for scene_info in scene_infos_cogs:
        if scene_info["scene_id"] in scene_infos_cogsafe_grouped.keys():
            scene_infos_cogsafe_grouped[scene_info["scene_id"]] += [scene_info]
        else:
            scene_infos_cogsafe_grouped[scene_info["scene_id"]] = [scene_info]

    scene_infos_cogsafe_condensed = []
    for scene_id, scene_info_list in scene_infos_cogsafe_grouped.items():
        if len(scene_info_list) == 1:
            scene_infos_cogsafe_condensed.append(scene_info_list[0])
        else:
            latest_scene_info = find_latest_sceneinfo(scene_info_list)
            scene_infos_cogsafe_condensed.append(latest_scene_info)

    # create final list with unique safe containers and cog
    scene_infos = scene_infos_safe_condensed
    for scene_cog_info in scene_infos_cogsafe_condensed:
        cog_scene_inseceinfos = False
        for scene_info_safe in scene_infos_safe_condensed:
            if scene_info_safe['scene_id'] == scene_cog_info['scene_id']:
                cog_scene_inseceinfos = True
        if not cog_scene_inseceinfos:
            print(f"adding {scene_cog_info['title']} to scenes")
            scene_infos.append(scene_cog_info)


    print("Final list")
    for scene_info in scene_infos:
        print(f'{scene_info["title"]}')
    return scene_infos