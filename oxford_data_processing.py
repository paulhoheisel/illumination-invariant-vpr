import json
import random
import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R_scipy
import time



def get_pose_dicts(json_path, day_image_names, night_image_names):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    day_pose_dict = {}
    night_pose_dict = {}
    for frame in data['frames']:
        full_path = frame['file_path']
        raw_name = full_path.split('/')[-1]  # "vrs00_59673489879.jpg"
        if '_' in raw_name:
            clean_name = raw_name.split('_', 1)[1]  # "59673489879.jpg"
        else:
            clean_name = raw_name
        
        # Die 4x4 Matrix (Camera-to-World)
        matrix = np.array(frame['transform_matrix'])

        R_w2c = matrix[0:3, 0:3]
        t_w2c = matrix[0:3, 3]
        
        # --- MATHEMATISCHE TRANSFORMATION (C2W) ---
        # Invertierung für das weltfeste Koordinatensystem
        R_c2w = R_w2c.T
        c_world = -R_c2w @ t_w2c

        # Convert rotation matrix to quaternion [qw, qx, qy, qz]
        R_quat = R_scipy.from_matrix(R_c2w)
        q_array = R_quat.as_quat()  # Returns [qx, qy, qz, qw]
        q = np.array([q_array[3], q_array[0], q_array[1], q_array[2]])  # Reorder to [qw, qx, qy, qz]
        
        t = c_world  # Translation in world coordinates
        
        if(clean_name in day_image_names):
            day_pose_dict[clean_name] = {
            'c_world': c_world,          
            'R_c2w': R_c2w,              
            'R_w2c_orig': R_w2c,
            't_w2c_orig': t_w2c,
            'q': q,
            't': t
            }
        
        elif(clean_name in night_image_names):
            night_pose_dict[clean_name] = {
            'c_world': c_world,          
            'R_c2w': R_c2w,              
            'R_w2c_orig': R_w2c,
            't_w2c_orig': t_w2c,
            'q': q,
            't': t
            }
        
    return day_pose_dict, night_pose_dict
            


def cluster_poses_with_matrices(dist_threshold=1.5, rot_weight=1.0, day_image_names=None, day_pose_dict=None):
    """
    translations: np.array (N, 3) -> [x, y, z]
    rotation_matrices: np.array (N, 3, 3) -> Die R-Matrizen
    rot_weight: Faktor, um den Einfluss der Rotation zu skalieren
    day_image_names: Liste der Bildnamen für die Tag-Poses
    """

    rotation_matrices = np.array([day_pose_dict[name]['R_c2w'] for name in day_image_names])
    translations = np.array([day_pose_dict[name]['c_world'] for name in day_image_names])

    # 1. Rotationsmatrizen flachklopfen (N, 9)
    R_flat = rotation_matrices.reshape(-1, 9)
    
    # 2. Features kombinieren: Translation (3) + Flattened R (9) = 12D Vektor
    # Wir skalieren R_flat, um das "Wackeln" der Aria-Brille im Clustering zu berücksichtigen
    combined_features = np.hstack([translations, R_flat * rot_weight])
    
    # 3. Clustering via BallTree (effizient für 12D)
    tree = BallTree(combined_features, leaf_size=40)
    
    visited = np.zeros(len(translations), dtype=bool)
    labels = {name: None for name in day_image_names}  # Bildname -> Cluster-ID
    cluster_id = 0
    
    # old "greedy" clustering that can overwrite clusters if a later pose is close to an already clustered one
    #for i in range(len(day_image_names)):
    #    if not visited[i]:
    #        # Suche alle Posen im gewichteten 12D-Radius
    #        indices = tree.query_radius(combined_features[i:i+1], r=dist_threshold)[0]
    ##        for idx in indices:
    #            labels[day_image_names[idx]] = cluster_id
    #        visited[indices] = True
    #        cluster_id += 1


    # non-overwriting clustering: only assign clusters to unlabeled images
    for i in range(len(day_image_names)):
        if not visited[i]:
            # Query radius around the unvisited anchor image
            indices = tree.query_radius(combined_features[i:i+1], r=dist_threshold)[0]
            
            # FILTER: Only take indices that have NOT been visited yet
            valid_indices = [idx for idx in indices if not visited[idx]]
            
            for idx in valid_indices:
                labels[day_image_names[idx]] = cluster_id
                
            visited[valid_indices] = True
            cluster_id += 1
            
    return labels


def find_best_day_night_pairs(poses_night, poses_day, dist_threshold=2, angle_threshold=10.0, day_labels=None, subroutes=None):
    """
    poses_night/day: Dict mit {'img_name': (rotation_matrix, translation_vec)}
    returns array containing (<name of night image>, <name of day image>, distance, angle difference)
    
    # 1. Daten für KD-Tree vorbereiten
    #if subroutes is not None:
    #    day_image_names = list()
        valid_labels = []
        for key in subroutes.keys():
            for j in range(len(subroutes[key])):
                valid_labels.append(subroutes[key][j])
        for img in poses_day.keys():
            if day_labels[img] in valid_labels:
                day_image_names.append(img)
    else:
        day_image_names = list(poses_day.keys())
    """
    day_image_names = list(poses_day.keys())

    night_image_names = list(poses_night.keys())
    
    t_night = np.array([poses_night[n]['c_world'] for n in night_image_names])
    t_day = np.array([poses_day[d]['c_world'] for d in day_image_names])
    
    # 2. KD-Tree auf Tag-Positionen aufbauen
    tree = KDTree(t_day)
    
    # Suche die k=50 nächsten Nachbarn für jedes Nacht-Bild
    dists, indices = tree.query(t_night, k=200, distance_upper_bound=dist_threshold)
    
    pairs = []
    
    # 3. Rotation auf CPU prüfen (Vektorisiert)
    device = torch.device("cpu")  # Use CPU to avoid CUDA issues
    
    for i, query_name in enumerate(night_image_names):
        valid_idx = indices[i][indices[i] < len(day_image_names)] # Filter Out-of-bounds
        if len(valid_idx) == 0: continue
        
        R_q = torch.from_numpy(poses_night[query_name]['R_c2w']).to(device)
        
        best_angle = float('inf')
        best_match = None
        
        for idx in valid_idx:
            day_name = day_image_names[idx]
            R_db = torch.from_numpy(poses_day[day_name]['R_c2w']).to(device)
            
            # Relative Rotation
            R_rel = torch.mm(R_q.t(), R_db)
            trace = torch.trace(R_rel)
            angle = torch.rad2deg(torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)))
            
            if angle < angle_threshold and angle < best_angle:
                best_angle = angle
                best_match = day_name 
            
        if best_match:
            pairs.append((query_name, best_match, dists[i][0], best_angle.item(), day_labels[best_match] if day_labels else None))
            
    return pairs


def get_image_names(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    image_names = []
    for frame in data['frames']:
        full_path = frame['file_path']
        raw_name = full_path.split('/')[-1]  # "vrs00_59673489879.jpg"
        if '_' in raw_name:
            clean_name = raw_name.split('_', 1)[1]  # "59673489879.jpg"
        else:
            clean_name = raw_name
        image_names.append(clean_name)
    return image_names


def cluster_poses_via_thresholds(rot_threshold=5.0, dist_threshold=1, day_image_names=None, day_pose_dict=None):
    '''
    Cluster poses based on distance and rotation thresholds.
    For each unvisited pose, create a new cluster and assign all poses within the thresholds to that cluster.
    This method is more interpretable and directly uses the defined thresholds, but can be less efficient for large datasets.
    returns:
        - labels: Dict mapping image names to cluster IDs
        - cluster_centers: List of image names that serve as cluster centers (the first image that started each cluster)
    '''
    not_visited = set(day_image_names)
    labels = {name: None for name in day_image_names}
    cluster_id = 0
    cluster_centers = []
    while not_visited:
        current_name = not_visited.pop()
        cluster_centers.append(current_name)
        labels[current_name] = cluster_id
        current_translation = day_pose_dict[current_name]['c_world']
        current_rotation = day_pose_dict[current_name]['R_c2w']
        
        to_visit = set()
        for other_name in not_visited:
            translation = day_pose_dict[other_name]['c_world']
            rotation = day_pose_dict[other_name]['R_c2w']
            
            dist = np.linalg.norm(translation - current_translation)
            angle = np.arccos((np.trace(current_rotation.T @ rotation) - 1) / 2) * (180.0 / np.pi)
            
            if dist < dist_threshold and angle < rot_threshold:
                to_visit.add(other_name)
        
        for name in to_visit:
            not_visited.remove(name)
            labels[name] = cluster_id

        
        cluster_id += 1
    
    return labels, cluster_centers, cluster_id


def construct_subroutes(labels, cluster_centers, day_pose_dict, dist_threshold, angle_threshold):
    '''
    Construct subroutes based on the clustered poses.
    For each cluster center, we check if it can be added to an existing subroute based on the distance and angle thresholds.
     - If it can be added to an existing subroute, we add it there.
     - If it cannot be added to any existing subroute, we start a new subroute with this cluster center as the first element.
     - The distance and angle are calculated between the current cluster center and the centers of the existing subroutes.
     - The function returns a dictionary where each key is a subroute ID and the value is a list of cluster_ids that belong to that subroute.
    '''
    subroutes = {}
    num_of_subroutes = 0
    visited_subroute_centers = set()
    for center_a in cluster_centers:
        feasible = False
        if center_a in visited_subroute_centers:
            continue
        visited_subroute_centers.add(center_a)
        current_translation = day_pose_dict[center_a]['c_world']
        current_rotation = day_pose_dict[center_a]['R_c2w']
        for subroute in subroutes.keys():
            feasible = True
            for loc in subroutes[subroute]:
                center_b = cluster_centers[loc]
                translation = day_pose_dict[center_b]['c_world']
                rotation = day_pose_dict[center_b]['R_c2w']
                if np.linalg.norm(translation - current_translation) < dist_threshold and np.arccos((np.trace(current_rotation.T @ rotation) - 1) / 2) * (180.0 / np.pi) < angle_threshold:
                    feasible = False
                    break
            if feasible:
                subroutes[subroute].append(labels[center_a])
                break
        if not feasible:
            subroutes[num_of_subroutes] = [labels[center_a]]
            num_of_subroutes += 1
    return subroutes



def filter_labels_to_pairs(day_labels, night_labels, pairs, subroutes=None, cluster_center_labeled_dict=None, train_scenes=None):

    num_labels = np.zeros(len(train_scenes), dtype=int)
    for i in range(len(train_scenes)):
    # Filtere die Elemente aus day_labels und night_labels heraus, die nicht zu einem cluster gehoeren, das in pairs enthalten ist
        print(f"anz day labels vorher: {len(set(day_labels[i].values()))}")
        day_labels[i] = {k: v for k, v in day_labels[i].items() if v in set(night_labels[i].values())}
        print(f"anz day labels nachher: {len(set(day_labels[i].values()))}")
        num_labels[i] = len(set(night_labels[i].values()))

        # aendere die labels, sodass sie von (0, NUM_CLASSES) gehen

        # build cluster assignment
        cluster_assignment = {}
        new_label = 0
        for old_label in set(night_labels[i].values()):
            cluster_assignment[old_label] = new_label
            new_label += 1
        if i == 3:
            print(f"cluster_assignment values: {cluster_assignment.values()}")

        # assign clusters in day_labels and night_labels
        temporary_day_labels = day_labels[i].copy()
        temporary_night_labels = night_labels[i].copy()
        for k, v in day_labels[i].items():
            temporary_day_labels[k] = cluster_assignment[v]
        for k, v in night_labels[i].items():
            temporary_night_labels[k] = cluster_assignment[v]
        day_labels[i] = temporary_day_labels
        night_labels[i] = temporary_night_labels

        # assign clusters in pairs
        for j in range(len(pairs[i])):
            d,n,dist,angle,label = pairs[i][j]
            pairs[i][j] = d,n,dist,angle,cluster_assignment[label]

        # assign clusters in subroutes
        if subroutes is not None:
            k = 0
            temporary_subroutes = {}
            for subroute in subroutes[i].keys():
                if len(subroutes[i][subroute]) == 0:
                    continue
                temporary_subroutes[k] = [
                    cluster_assignment[label]
                    for label in subroutes[i][subroute]
                    if label in cluster_assignment.keys() 
                ]
                k += 1
            subroutes[i] = temporary_subroutes

        # assign clusters in cluster_center_labeled_dict
        if cluster_center_labeled_dict is not None:
            new_cluster_centers = {cluster_assignment[label]: center for (label, center) in cluster_center_labeled_dict[i].items() if label in cluster_assignment.keys()}
            print(new_cluster_centers)
            (print(len(new_cluster_centers)))
            cluster_center_labeled_dict[i] = new_cluster_centers
            print(f"lenght of new cluster_center_labeled_dict[i]: {len(cluster_center_labeled_dict[i])}")
    return day_labels, night_labels, pairs, num_labels, subroutes, cluster_center_labeled_dict



def combine_labels_across_scenes(day_labels, night_labels, pairs, num_labels, subroutes=None, cluster_center_labeled_dict=None, train_scenes=None):
    # Combine all pairs into one list for training and adjust labels, so labels from a different scene dont have the same label (e.g. cluster 0 from scene 1 and cluster 0 from scene 2 should get different labels, because they are different places)
    label_offset = 0
    for i in range(len(train_scenes)):
        for j in range(len(pairs[i])):
            d,n,dist,angle,label = pairs[i][j]
            pairs[i][j] = (d, n, dist, angle, label + label_offset)
        day_labels[i] = {k: v + label_offset for k, v in day_labels[i].items()}
        night_labels[i] = {k: v + label_offset for k, v in night_labels[i].items()}

        updated_cluster_centers = {k + label_offset: v for k, v in cluster_center_labeled_dict[i].items()}
        cluster_center_labeled_dict[i] = updated_cluster_centers
          
        # change labels in subroutes
        if subroutes is not None:
            for subroute in subroutes[i].keys():
                subroutes[i][subroute] = [label + label_offset for label in subroutes[i][subroute]]
                
        label_offset += num_labels[i] 
    NUM_TOTAL_CLASSES = label_offset

    # old version
    # combine subroutes to one dictionary with global labels and remove empty subroutes
    #if subroutes is not None:
    #    combined_subroutes = {}
    #    counter = 0
    #    for i in range(len(train_scenes)):
    #        for subroute in subroutes[i].keys():
    #            if len(subroutes[i][subroute]) > 0:  # Nur nicht-leere Subrouten hinzufügen
    #                combined_subroutes[counter] = subroutes[i][subroute]
    #                counter += 1
    #    subroutes = combined_subroutes

    if subroutes is not None:
        number_of_subroutes_per_scene = len(subroutes[0])
        for i in range(len(subroutes)):
            if len(subroutes[i]) != number_of_subroutes_per_scene:
                raise ValueError("Not all scenes have the same number of subroutes!")
            
        combined_subroutes = {}
        for i in range(number_of_subroutes_per_scene):
            combined_subroutes[i] = []
            for j in range(len(subroutes)):
                combined_subroutes[i].extend(subroutes[j][i])
        subroutes=combined_subroutes
    print(f"Total number of classes across all scenes: {NUM_TOTAL_CLASSES}")

    # make day_labels and night_labels flattened across scenes
    day_labels_flat = {}
    night_labels_flat = {}
    cluster_center_labeled_dict_flat = {}
    for i in range(len(train_scenes)):
        day_labels_flat.update(day_labels[i])
        night_labels_flat.update(night_labels[i])
        # flatten cluster_center_labeled_dict
        cluster_center_labeled_dict_flat.update(cluster_center_labeled_dict[i])
    
    return day_labels_flat, night_labels_flat, pairs, NUM_TOTAL_CLASSES, subroutes, cluster_center_labeled_dict_flat





def build_optimized_subroutes(day_image_names, day_pose_dict, 
                              dist_threshold=0.5, 
                              rot_threshold=5.0, 
                              min_imgs_per_cluster=10, 
                              max_imgs_per_cluster=30,
                              num_subroutes=10):
    """
    Cluster poses efficiently using KD-Trees, sorts them spatially, 
    applies intra-cluster pruning, and assigns interleaved subroutes.
    
    Returns:
        - labels: Dict mapping image_name -> final_cluster_id
        - subroutes_dict: Dict mapping subroute_id -> list of final_cluster_ids
        - cluster_centers_dict: Dict mapping final_cluster_id -> center_image_name
    """
    N = len(day_image_names)
    names_array = np.array(day_image_names)
    
    # 1. Daten extrahieren und in flache Numpy-Arrays konvertieren
    translations = np.empty((N, 3))
    rotations = np.empty((N, 3, 3))
    
    for i, name in enumerate(day_image_names):
        translations[i] = day_pose_dict[name]['c_world']
        rotations[i] = day_pose_dict[name]['R_c2w']
        
    print("Building KD-Tree...")
    tree = cKDTree(translations)
    
    visited = np.zeros(N, dtype=bool)
    raw_clusters = []
    cluster_centers_pos = []
    raw_centers_names = []  # NEU: Trackt den Namen des Center-Bildes
    
    print("Clustering images...")
    for i in range(N):
        if visited[i]: 
            continue
            
        idx_candidates = tree.query_ball_point(translations[i], dist_threshold)
        idx_candidates = [idx for idx in idx_candidates if not visited[idx]]
        
        if not idx_candidates:
            continue
            
        # Vektorisierter Batch-Winkel-Check
        R_current = rotations[i]
        R_cands = rotations[idx_candidates]
        
        traces = np.sum(R_current * R_cands, axis=(1, 2))
        traces = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
        angles = np.arccos(traces) * (180.0 / np.pi)
        
        valid_mask = angles < rot_threshold
        valid_indices = np.array(idx_candidates)[valid_mask]
        
        visited[valid_indices] = True
        raw_clusters.append([names_array[idx] for idx in valid_indices])
        cluster_centers_pos.append(translations[i])
        raw_centers_names.append(day_image_names[i])  # NEU: Das aktuelle Bild i startet das Cluster

    print(f"Initial clusters formed: {len(raw_clusters)}")

    # 4. Klassen-Pruning (Zu kleine Cluster verwerfen)
    valid_clusters = []
    valid_centers_pos = []
    valid_centers_names = []  # NEU: Filtert die Center-Namen analog
    for cluster, center_pos, center_name in zip(raw_clusters, cluster_centers_pos, raw_centers_names):
        if len(cluster) >= min_imgs_per_cluster:
            valid_clusters.append(cluster)
            valid_centers_pos.append(center_pos)
            valid_centers_names.append(center_name)
            
    print(f"Clusters after pruning (<{min_imgs_per_cluster} imgs removed): {len(valid_clusters)}")
    
    if not valid_clusters:
        return {}, {}, {}
        
    # 5. Räumliche Sortierung (Greedy Nearest Neighbor)
    sorted_clusters = []
    sorted_centers_names = []  # NEU: Speichert die Center-Namen in der sortierten Reihenfolge
    
    K = len(valid_centers_pos)
    centers_array = np.array(valid_centers_pos)
    visited_centers = np.zeros(K, dtype=bool)
    
    current_idx = 0
    visited_centers[current_idx] = True
    sorted_clusters.append(valid_clusters[current_idx])
    sorted_centers_names.append(valid_centers_names[current_idx])
    
    for _ in range(1, K):
        dists = np.linalg.norm(centers_array - centers_array[current_idx], axis=1)
        dists[visited_centers] = np.inf
        
        next_idx = np.argmin(dists)
        visited_centers[next_idx] = True
        sorted_clusters.append(valid_clusters[next_idx])
        sorted_centers_names.append(valid_centers_names[next_idx])  # NEU: Synchron mitwandern
        current_idx = next_idx

    # 6. Intra-Cluster Filtering (Uniform Downsampling)
    final_clusters = []
    for cluster in sorted_clusters:
        if len(cluster) > max_imgs_per_cluster:
            indices = np.linspace(0, len(cluster)-1, max_imgs_per_cluster, dtype=int)
            filtered_cluster = [cluster[idx] for idx in indices]
            final_clusters.append(filtered_cluster)
        else:
            final_clusters.append(cluster)

    # 7. Labels, Subrouten und Center-Mapping generieren
    labels = {}
    subroutes_dict = {i: [] for i in range(num_subroutes)}
    cluster_centers_dict = {}  # NEU: Das finale Ausgabe-Dictionary
    
    for new_cluster_id, cluster_images in enumerate(final_clusters):
        subroute_id = new_cluster_id % num_subroutes
        subroutes_dict[subroute_id].append(new_cluster_id)
        
        # NEU: Zuweisung des exakten Center-Namens für die neue, finale ID
        cluster_centers_dict[new_cluster_id] = sorted_centers_names[new_cluster_id]
        
        for img_name in cluster_images:
            labels[img_name] = new_cluster_id
            
    print(f"Successfully generated {len(final_clusters)} classes distributed over {num_subroutes} subroutes.")
            
    return labels, subroutes_dict, cluster_centers_dict





def find_best_day_night_pairs_optimized(poses_night, poses_day, day_labels, 
                                       dist_threshold=0.5, angle_threshold=5.0):
    """
    Findet die besten Tag-Bilder für die Nacht-Bilder basierend auf den gefilterten Clustern.
    Nutzt hocheffiziente KD-Tree Queries und vektorisierte Rotationsberechnungen.
    
    Inputs:
        - poses_night: Dict mit Nacht-Posen {'img_name': {'c_world': ..., 'R_c2w': ...}}
        - poses_day: Dict mit Tag-Posen (vollständiges Set)
        - day_labels: Dict der gültigen Tag-Bilder gemappt auf ihre finale Cluster-ID
        - dist_threshold: Maximaler räumlicher Abstand (z.B. 0.5m für Handheld)
        - angle_threshold: Maximaler Winkelunterschied (z.B. 5.0° für Handheld)
        
    Returns:
        - pairs: Liste von Tuples: (night_img, day_img, distance, angle_diff, cluster_id)
    """
    # 1. Nur Tag-Bilder behalten, die das Clustering/Pruning überlebt haben
    valid_day_names = [name for name in poses_day.keys() if name in day_labels]
    if not valid_day_names:
        print("Error: Keine Übereinstimmung zwischen poses_day und day_labels gefunden!")
        return []
    
    valid_day_names_arr = np.array(valid_day_names)
    
    # Tag-Arrays für die Vektorisierung extrahieren
    t_day = np.array([poses_day[name]['c_world'] for name in valid_day_names])
    R_day = np.array([poses_day[name]['R_c2w'] for name in valid_day_names])
    
    # Nacht-Arrays extrahieren
    night_image_names = list(poses_night.keys())
    t_night = np.array([poses_night[name]['c_world'] for name in night_image_names])
    R_night = np.array([poses_night[name]['R_c2w'] for name in night_image_names])
    
    # 2. KD-Tree auf den gefilterten Tag-Positionen aufbauen
    tree = cKDTree(t_day)
    
    # Alle Tag-Kandidaten im Umkreis für ALLE Nacht-Bilder gleichzeitig suchen
    print("Querying KD-Tree for all night frames...")
    all_candidate_indices = tree.query_ball_point(t_night, dist_threshold)
    
    pairs = []
    
    print("Running vectorized angular filtering...")
    # 3. Iteration über die Nacht-Bilder (die innere Schleife ist komplett vektorisiert)
    for i, night_name in enumerate(night_image_names):
        cand_indices = all_candidate_indices[i]
        if not cand_indices: 
            continue  # Kein Tag-Bild im räumlichen Umkreis
            
        # Extrahiere Rotationen aller Tag-Kandidaten für dieses eine Nacht-Bild
        R_day_cands = R_day[cand_indices]  # Shape: (Anzahl_Kandidaten, 3, 3)
        R_q = R_night[i]                   # Shape: (3, 3)
        
        # --- VEKTORISIERTER BATCH-WINKEL-CHECK ---
        # tr(R_q^T @ R_day) entspricht der Elementweisen Multiplikation und Summation über die Achsen
        traces = np.sum(R_q * R_day_cands, axis=(1, 2))
        traces = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
        angles = np.arccos(traces) * (180.0 / np.pi)
        
        # Filter nach deinem harten Winkel-Threshold
        valid_mask = angles < angle_threshold
        if not np.any(valid_mask):
            continue
            
        # Wie in deiner alten Methode: Wähle das Tag-Bild mit dem kleinsten Winkel
        valid_cand_indices = np.array(cand_indices)[valid_mask]
        valid_angles = angles[valid_mask]
        
        best_idx_in_mask = np.argmin(valid_angles)
        best_day_idx = valid_cand_indices[best_idx_in_mask]
        best_angle = valid_angles[best_idx_in_mask]
        
        best_day_name = valid_day_names_arr[best_day_idx]
        
        # Exakte euklidische Distanz berechnen
        exact_dist = np.linalg.norm(t_night[i] - t_day[best_day_idx])
        
        # Cluster ID aus deinem bereinigten day_labels Dict holen
        cluster_id = day_labels[best_day_name]
        
        # Struktur exakt wie in deiner alten Methode beibehalten
        pairs.append((
            night_name, 
            best_day_name, 
            float(exact_dist), 
            float(best_angle), 
            cluster_id
        ))
        
    print(f"Successfully generated {len(pairs)} day-night pairs.")
    return pairs

def filter_subroutes(subroutes, min_size):
    filtered_subroutes = {s: subroutes[s] for s in range(len(subroutes)) if len(subroutes[s]) >= min_size}
    return filtered_subroutes


def filter_pose_dict(used_images, dict):
    filtered_dict = {img: dict[img] for img in used_images.keys()}
    return filtered_dict


def visualize_cluster_centers_2d(cluster_centers_dict_flat, day_dict_flat, scene_number = 0, num_classes = [0,0,0,0]):
    """
    Test: Visualize cluster centers in a 2D plane (ignoring elevation).
    
    Args:
        cluster_centers_dict_flat: Dict mapping cluster_id -> image_name (cluster center)
        day_dict_flat: Dict mapping image_name -> {'c_world': [x, y, z], ...}
    
    Returns:
        fig, ax: matplotlib figure and axes objects
    """

    min_cluster_id = 0
    max_cluster_id = 0
    for i in range(scene_number):
        min_cluster_id += num_classes[i]
    for i in range(scene_number + 1):
        max_cluster_id += num_classes[i]

    # Extract 2D coordinates for each cluster center
    cluster_coords = {}
    cluster_ids = []
    x_coords = []
    y_coords = []
    
    for cluster_id, image_name in cluster_centers_dict_flat.items():
        if image_name in day_dict_flat and cluster_id >= min_cluster_id and cluster_id <= max_cluster_id:
            pose_data = day_dict_flat[image_name]
            c_world = pose_data['c_world']  # [x, y, z]
            
            cluster_coords[cluster_id] = c_world[[0, 1]]  # Extract x, y (ignore z)
            cluster_ids.append(cluster_id)
            x_coords.append(c_world[0])
            y_coords.append(c_world[1])
    
    print(f"Visualizing {len(cluster_ids)} cluster centers")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize cluster IDs to [0, 1] for colormap
    if len(cluster_ids) > 0:
        min_id = min(cluster_ids)
        print(min_id)
        max_id = max(cluster_ids)
        print(max_id)
        normalized_ids = [(cid - min_id) / (max_id - min_id + 1) if max_id > min_id else 0.5 
                          for cid in cluster_ids]
        
        # Use a colormap: light colors for low IDs, dark colors for high IDs
        cmap = get_cmap('viridis')  # Light to dark
        colors = [cmap(norm_id) for norm_id in normalized_ids]
        
        # Scatter plot
        scatter = ax.scatter(x_coords, y_coords, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add cluster labels on the plot
        for i, cluster_id in enumerate(cluster_ids):
            ax.annotate(f'{cluster_id}', 
                       xy=(x_coords[i], y_coords[i]), 
                       xytext=(3, 3), 
                       textcoords='offset points',
                       fontsize=7,
                       alpha=0.6)
    
    ax.set_xlabel('X coordinate (meters)', fontsize=12)
    ax.set_ylabel('Y coordinate (meters)', fontsize=12)
    ax.set_title(f'2D Spatial Distribution of {len(cluster_ids)} Cluster Centers\n(Light = Low Cluster Numbers, Dark = High Cluster Numbers)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_id if len(cluster_ids) > 0 else 0, 
                                                              vmax=max_id if len(cluster_ids) > 0 else 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Cluster ID')
    
    plt.tight_layout()
    
    return fig, ax, cluster_coords


def build_path_mapping(image_directory, train_scenes=None):
    path_mapping = {}
    
    # Durchsucht rekursiv alle Unterordner nach jpgs (oder pngs)
    for i in range(len(train_scenes)):
        for file_path in Path(image_directory[i]).rglob("*.jpg"):
            filename = file_path.name
            
            # Schneide das Präfix ab (gleiche Logik wie bei der JSON-Extraktion)
            if '_' in filename:
                short_key = filename.split('_', 1)[1]
            else:
                short_key = filename
                
            path_mapping[short_key] = str(file_path)
        
    return path_mapping



def _quat_to_rotmat(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def _rotation_angle_deg(q1, q2):
    R1 = _quat_to_rotmat(q1)
    R2 = _quat_to_rotmat(q2)
    R_rel = R1.T @ R2
    trace = np.trace(R_rel)
    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(angle_rad)


class ThresholdPairDataset(Dataset):
    def __init__(self, dist_thresholds, angle_thresholds, poses_dict_1, poses_dict_2, image_names_1, image_names_2,
                 path_map_1, path_map_2, sample_size, transform=None, build_pairs=True):
        """
        Erzeugt Bildpaare pro Threshold-Gruppe, wobei jedes Paar zufällig aus allen
        Kandidaten gezogen wird, die zwischen dem aktuellen und dem vorherigen threshold liegen.

        Jedes Paar wird mit einem Threshold-Label annotiert, das für das Training und
        die spätere Zuordnung über den DataLoader geeignet ist.
        """
        self.dist_thresholds = np.array(dist_thresholds)
        self.angle_thresholds = np.array(angle_thresholds)
        self.poses_dict_1 = poses_dict_1
        self.poses_dict_2 = poses_dict_2
        self.image_names_1 = list(image_names_1)
        self.image_names_2 = list(image_names_2)
        self.path_map_1 = path_map_1
        self.path_map_2 = path_map_2
        self.sample_size = int(sample_size)

        self.pairs_by_threshold = {}
        self.threshold_labels = {}
        self.pairs = []

        self.transform = transform or T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if build_pairs:
            self.build_pairs()

    
    def build_pairs(self):
        eps = 1e-2
        rng = random.Random(42)
        print("Threshold-based pair generation started:")

        # Prepare numpy arrays + quaternions for fast queries
        names1 = list(self.image_names_1)
        names2 = list(self.image_names_2)
        t1s = np.array([np.asarray(self.poses_dict_1[n]['t'], dtype=np.float32) for n in names1])
        t2s = np.array([np.asarray(self.poses_dict_2[n]['t'], dtype=np.float32) for n in names2])
        q1s = [self.poses_dict_1[n]['q'] for n in names1]
        q2s = [self.poses_dict_2[n]['q'] for n in names2]

        # KDTree on second set to quickly find spatial neighbors
        if len(t2s) > 0:
            tree2 = KDTree(t2s)
        else:
            tree2 = None

        label_idx = 0
        for ind_d in range(len(self.dist_thresholds)):
            for ind_a in range(len(self.angle_thresholds)):
                d, a = self.dist_thresholds[ind_d], self.angle_thresholds[ind_a]
                if ind_d > 0:
                    d_previous = self.dist_thresholds[ind_d-1]
                else: d_previous = 0
                if ind_a > 0:
                    a_previous = self.angle_thresholds[ind_a-1]
                else: a_previous = 0
                collected = []
                seen = set()

                # Randomize scanning order of first set to avoid spatial bias
                indices1 = list(range(len(names1)))
                rng.shuffle(indices1)

                # Phase 1: spatial neighbor based search
                if tree2 is not None:
                    for i in indices1:
                        if len(collected) >= self.sample_size:
                            break
                        neighbors = tree2.query_ball_point(t1s[i], r=d)
                        if not neighbors:
                            print("keine neighbors gefunden")
                            continue
                        #print(f"{len(neighbors)} neighbors gefunden")
                        rng.shuffle(neighbors)
                        for j in neighbors:
                            key = (names1[i], names2[j])
                            if key in seen:
                                continue
                            dist = float(np.linalg.norm(t2s[j] - t1s[i]))
                            angle = float(_rotation_angle_deg(q1s[i], q2s[j]))
                            if d_previous <= dist <= d and a_previous <= angle <= a:
                                if dist < eps and angle < eps:
                                    continue
                                pair = {
                                    'img_name_1': names1[i],
                                    'img_name_2': names2[j],
                                    'path_1': self.path_map_1[names1[i]],
                                    'path_2': self.path_map_2[names2[j]],
                                    'threshold': (d, a),
                                    'threshold_label': label_idx,
                                    'distance': dist,
                                    'angle': angle,
                                }
                                #print("added pair via tree")
                                collected.append(pair)
                                seen.add(key)
                                break
                    label_idx += 1

                # Phase 2: fallback random sampling until we have enough or give up
                attempts = 0
                if len(collected) < self.sample_size:
                    print(f"fallback...after finding only {len(collected)} pairs.")
                max_attempts = max(1000, self.sample_size * 50)
                while len(collected) < self.sample_size and attempts < max_attempts:
                    i = rng.randrange(len(names1))
                    j = rng.randrange(len(names2))
                    key = (names1[i], names2[j])
                    attempts += 1
                    if key in seen:
                        continue
                    dist = float(np.linalg.norm(t2s[j] - t1s[i]))
                    if dist > d:
                        continue
                    angle = float(_rotation_angle_deg(q1s[i], q2s[j]))
                    if angle > a:
                        continue
                    if dist <= d_previous or angle <= a_previous:
                        continue
                    if dist < eps and angle < eps:
                        continue
                    pair = {
                        'img_name_1': names1[i],
                        'img_name_2': names2[j],
                        'path_1': self.path_map_1[names1[i]],
                        'path_2': self.path_map_2[names2[j]],
                        'threshold': (d, a),
                        'threshold_label': label_idx,
                        'distance': dist,
                        'angle': angle,
                    }
                    collected.append(pair)
                    seen.add(key)

                # Save
                self.pairs_by_threshold[(d, a)] = collected
                for pair in collected:
                    self.pairs.append(pair)

                self.threshold_labels[(d, a)] = label_idx
                print(f"  threshold {label_idx}: {(d, a)} -> created {len(collected)} pairs")
                if len(collected) < self.sample_size:
                    print(f"  warning: threshold {(d, a)} did not reach sample_size ({len(collected)} < {self.sample_size})")

        return self.pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        img_1 = Image.open(pair['path_1']).convert('RGB')
        img_2 = Image.open(pair['path_2']).convert('RGB')

        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return img_1, img_2, pair['threshold_label']


def viewpoint_invariance_validation(dataset, night_expert=None,
                                    criterion=(0.0, 10.0, 0.0, 15.0),
                                    device=None,
                                    dataloader=None,
                                    batch_size=128,
                                    num_workers=4,
                                    pin_memory=False,
                                    plot=False,
                                    fit_surface=False):
    """
    Validate viewpoint invariance on a thresholded pair dataset.
    
    This function creates the criterion mask first and only computes embeddings
    for pairs within the criterion range, improving efficiency.

    Args:
        dataset: ThresholdPairDataset with precomputed image pairs.
        night_expert: Model that maps images to embedding vectors.
            If None, tries to use a globally visible `night_expert` variable.
        criterion: Tuple (min_dist, max_dist, min_angle, max_angle).
        device: Torch device to run the model on.
        dataloader: Optional pre-built DataLoader. If None, creates one from dataset.
        batch_size: Batch size used during evaluation.
        num_workers: Number of DataLoader workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        plot: Whether to create a 3D scatter plot of results.
        fit_surface: If True, fits a 2D polynomial surface to the data and displays it on the plot.

    Returns:
        average_similarity: Average cosine similarity for pairs within the criterion range.
        results: List of (distance, angle, cosine_similarity) tuples for selected pairs only.
    """

    print("Evaluating viewpoint variance on validation set...")
    start = time.time()

    if night_expert is None:
        night_expert = globals().get('night_expert', None)
    if night_expert is None:
        raise ValueError('night_expert must be provided or defined globally')

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(criterion, dict):
        min_dist = criterion.get('min_dist', 0.0)
        max_dist = criterion.get('max_dist', float('inf'))
        min_angle = criterion.get('min_angle', 0.0)
        max_angle = criterion.get('max_angle', float('inf'))
        print("Warning: angle and dist criterion was not provided properly!")
    elif len(criterion) == 4:
        min_dist, max_dist, min_angle, max_angle = criterion
    else:
        raise ValueError('criterion must be a 4-tuple or dict')

    min_dist = max(0.0, min_dist)
    max_dist = max_dist
    min_angle = max(0.0, min_angle)
    max_angle = max_angle

    print(f"min dist: {min_dist}, max dist: {max_dist}, min angle: {min_angle}, max_angle:{max_angle}")

    night_expert = night_expert.to(device)
    night_expert.eval()

    # Pre-extract distance/angle metadata in dataset order (EARLY - BEFORE computing anything)
    distances = np.array([float(p.get('distance', 0.0)) for p in getattr(dataset, 'pairs', [])], dtype=np.float32)
    angles = np.array([float(p.get('angle', 0.0)) for p in getattr(dataset, 'pairs', [])], dtype=np.float32)

    # CREATE MASK FIRST based on criterion (before computing embeddings)
    mask = (
        (distances >= min_dist) & (distances <= max_dist) &
        (angles >= min_angle) & (angles <= max_angle)
    )
    
    selected_indices = np.where(mask)[0]
    print(f"Number of pairs within criterion: {np.sum(mask)} / {len(distances)}")

    # Create a filtered dataset containing only selected indices
    class IndexedSubset(torch.utils.data.Subset):
        def __getitem__(self, idx):
            actual_idx = self.indices[idx]
            return self.dataset[actual_idx]

    filtered_dataset = IndexedSubset(dataset, selected_indices)
    
    # Build dataloader only for selected pairs
    if dataloader is None:
        dataloader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)

    # Compute embeddings only for selected pairs
    cosines_list = []
    with torch.no_grad():
        for batch in dataloader:
            # Expect batch to be (img1_batch, img2_batch, _)
            try:
                img_1_batch, img_2_batch, _ = batch
            except Exception:
                raise TypeError('Expected dataset to return (img1, img2, label_or_meta) per __getitem__')

            if not torch.is_tensor(img_1_batch) or not torch.is_tensor(img_2_batch):
                raise TypeError('Dataset __getitem__ must return torch.Tensor images for batching')

            img_1_batch = img_1_batch.to(device)
            img_2_batch = img_2_batch.to(device)

            feat_1 = night_expert(img_1_batch)
            feat_2 = night_expert(img_2_batch)

            if feat_1.ndim > 2:
                feat_1 = feat_1.view(feat_1.shape[0], -1)
            if feat_2.ndim > 2:
                feat_2 = feat_2.view(feat_2.shape[0], -1)

            batch_cos = F.cosine_similarity(feat_1, feat_2, dim=1).cpu().numpy()
            cosines_list.append(batch_cos)

    if cosines_list:
        cosines = np.concatenate(cosines_list, axis=0)
    else:
        cosines = np.array([], dtype=np.float32)

    # Build results array with ONLY selected pairs
    selected_distances = distances[selected_indices]
    selected_angles = angles[selected_indices]
    results_arr = np.stack([selected_distances, selected_angles, cosines], axis=1) if len(cosines) > 0 else np.empty((0, 3), dtype=np.float32)

    # Plot if requested
    coeffs = np.zeros(6)
    if plot:
        from mpl_toolkits.mplot3d import Axes3D
        from scipy.interpolate import griddata
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        if results_arr.shape[0] > 0:
            scatter = ax.scatter(results_arr[:, 0], results_arr[:, 1], results_arr[:, 2],
                                c=results_arr[:, 2], cmap='viridis', s=30, alpha=0.8, label='Data points')
            fig.colorbar(scatter, ax=ax, label='Cosine similarity')
            
            # Fit and plot surface if requested
            if fit_surface and len(results_arr) > 3:
                print("Fitting 2D polynomial surface to data...")
                
                # Prepare data for fitting
                x = results_arr[:, 0]  # distances
                y = results_arr[:, 1]  # angles
                z = results_arr[:, 2]  # cosines
                
                # Use 2D polynomial fitting (order 2 for a smooth surface)
                # Create a polynomial of the form: z = c0 + c1*x + c2*y + c3*x^2 + c4*y^2 + c5*x*y
                A = np.vstack([np.ones_like(x), x, y, x**2, y**2, x*y]).T
                coeffs, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
                
                # Create grid for surface
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                x_grid = np.linspace(x_min, x_max, 20)
                y_grid = np.linspace(y_min, y_max, 20)
                X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
                
                # Evaluate polynomial on grid
                Z_grid = (coeffs[0] + 
                         coeffs[1] * X_grid + 
                         coeffs[2] * Y_grid + 
                         coeffs[3] * X_grid**2 + 
                         coeffs[4] * Y_grid**2 + 
                         coeffs[5] * X_grid * Y_grid)
                
                # Clip Z values to reasonable range
                Z_grid = np.clip(Z_grid, -1.0, 1.0)
                
                # Plot surface with transparency
                surf = ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3, cmap='coolwarm', label='Fitted surface')
                
                print(f"Surface fitting complete. Polynomial coefficients: {coeffs}")
        
        ax.set_xlabel('Distance (m)', fontsize=11)
        ax.set_ylabel('Angle (degrees)', fontsize=11)
        ax.set_zlabel('Cosine similarity', fontsize=11)
        title_str = 'Viewpoint Invariance Validation'
        if fit_surface:
            title_str += ' (with fitted surface)'
        ax.set_title(title_str, fontsize=12, fontweight='bold')
        plt.tight_layout()

    # Compute average similarity for selected pairs only
    if np.any(mask) and len(cosines) > 0:
        average_similarity = float(np.mean(cosines))
    else:
        average_similarity = float('nan')

    results = [tuple(row.tolist()) for row in results_arr]

    return average_similarity, results, coeffs


class OxfordDayNightDataset(Dataset):
    def __init__(self, pairs, day_labels, path_map, transform=None, subroutes=None):
        """
        pairs: Liste von dicts [{'night_img': '123.jpg', 'day_img': '456.jpg'}, ...] Use new_pairs and not pairs!!
        day_labels: Dict mit den Cluster-Labels {'456.jpg': 12, ...}
        path_map: maps short path to full image path
        subroutes: Liste von Subrouten, wobei jede Subroute eine Liste von Cluster-IDs ist, die zu dieser Subroute gehören. z.B. [[0, 1, 2], [3, 4], ...]
        """
        # divide pairs into the subroutes, so that we can later sample from the subroutes during training
        if subroutes is not None:
            # Initialize pairs_by_subroute with the actual keys from subroutes dict, not range()
            pairs_by_subroute = {key: [] for key in subroutes.keys()}
            for pair in pairs:
                day_img = pair['day_img']
                day_label = day_labels[day_img]
                for subroute_id, cluster_ids in subroutes.items():
                    if day_label in cluster_ids:
                        pairs_by_subroute[subroute_id].append(pair)
                        break
            self.pairs_by_subroute = pairs_by_subroute
            print(f"pairs by subroute attribute created with {len(self.pairs_by_subroute)} subroutes.")
        self.pairs = pairs
        self.labels = day_labels
        self.path_map = path_map
        self.subroutes = subroutes
        if subroutes is not None:
            self.set_subroute(list(subroutes.keys())[0])  # Set to first subroute using actual keys

        # Standard-Transforms für ResNet (falls keine übergeben wurden)
        self.transform = transform or T.Compose([
            T.Resize((512, 512)), 
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        # Return length of current subroute if one is active
        if hasattr(self, 'current_pairs'):
            return len(self.current_pairs)
        elif hasattr(self, 'pairs_by_subroute'):
            return sum(len(pairs) for pairs in self.pairs_by_subroute.values())
        else:
            return len(self.pairs)
        
    def set_subroute(self, subroute_id):
        if hasattr(self, 'pairs_by_subroute') and subroute_id in self.pairs_by_subroute:
            self.current_pairs = self.pairs_by_subroute[subroute_id]
        else:
            raise ValueError(f"Ungültige Subroute-ID: {subroute_id}")

    def __getitem__(self, idx):
        if hasattr(self, 'pairs_by_subroute'):
            pair = self.current_pairs[idx]
        else:
            pair = self.pairs[idx]
        night_key = pair['night_img']
        day_key = pair['day_img']
        
        # 1. Reale Pfade nachschlagen
        night_path = self.path_map[night_key]
        day_path = self.path_map[day_key]
        
        # 2. Bilder laden (RGB, um Alpha-Kanäle etc. zu ignorieren)
        night_img = Image.open(night_path).convert('RGB')
        day_img = Image.open(day_path).convert('RGB')
        
        # 3. Tensoren erstellen
        if self.transform:
            night_img = self.transform(night_img)
            day_img = self.transform(day_img)
            
        # 4. Cluster-Label abrufen (Das Nachtbild erbt das Label des Tagbildes!)
        label = self.labels[day_key]
        
        return night_img, day_img, label
    



def create_scene_splits(root_dir, val_prefixes = ('1_1_', '2_1_', '3_1_', '4_1_'), test_prefixes = ('1_2_', '2_2_', '3_2_', '4_2_')):
    # Lies alle Bildnamen aus dem 'day' Ordner (da 'night' identisch benannt ist)
    day_dir = os.path.join(root_dir, 'Day')
    night_dir = os.path.join(root_dir, 'Night')
    
    train_pairs = []
    val_pairs = []
    test_pairs = []
    
    image_names = [f for f in os.listdir(day_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in image_names:
        
        # Split-Logik nach Präfixen
        if img_name.startswith(val_prefixes):
            val_pairs.append(img_name)
        elif img_name.startswith(test_prefixes):
            test_pairs.append(img_name)
        else:
            train_pairs.append(img_name)
        
    print(f"Assigned {len(train_pairs)} train pairs, {len(val_pairs)} val pairs and {len(test_pairs)} test pairs.")
    return train_pairs, val_pairs, test_pairs

# Anwendung:
# train_list, val_list, test_list = create_scene_splits("/pfad/zu/DarkDriving")
# print(f"Train: {len(train_list)} | Val: {len(val_list)} | Test: {len(test_list)}")


def illumination_invariance_validation(dark_driving_split, night_expert=None,
                                       day_expert=None, batch_size=128,
                                       num_workers=4, pin_memory=False,
                                       device=None):
    """Evaluate illumination invariance by comparing night/day embeddings.

    The function expects the provided dataset to yield pairs of tensors
    ``(day_image, night_image)`` for each sample. It runs the night expert on
    night images and the day expert on day images, then computes cosine
    similarity between the resulting embeddings.

    Args:
        dark_driving_split: Dataset-like object with ``__len__`` and ``__getitem__``.
        night_expert: Model producing embeddings for night images.
        day_expert: Model producing embeddings for day images.
        batch_size: Batch size used during evaluation.
        num_workers: Number of DataLoader workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        device: Torch device to run on. If None, uses CUDA when available.

    Returns:
        mean_similarity: Mean cosine similarity across all pairs.
        scores: List of cosine similarities in dataset order.
    """

    print("Evaluating illumination invariance on validation set...")

    if night_expert is None:
        night_expert = globals().get('night_expert', None)
    if day_expert is None:
        day_expert = globals().get('day_expert', None)

    if night_expert is None or day_expert is None:
        raise ValueError('night_expert and day_expert must be provided or defined globally')

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    night_expert = night_expert.to(device)
    day_expert = day_expert.to(device)
    night_expert.eval()
    day_expert.eval()

    dataloader = DataLoader(
        dark_driving_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    scores = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                day_batch, night_batch = batch
            elif len(batch) == 3:
                day_batch, night_batch, _ = batch
            else:
                raise TypeError('Expected dataset to return (day_image, night_image) or (day_image, night_image, label)')

            night_batch = night_batch.to(device)
            day_batch = day_batch.to(device)

            night_features = night_expert(night_batch)
            day_features = day_expert(day_batch)

            if night_features.ndim > 2:
                night_features = night_features.view(night_features.shape[0], -1)
            if day_features.ndim > 2:
                day_features = day_features.view(day_features.shape[0], -1)

            batch_scores = F.cosine_similarity(night_features, day_features, dim=1).cpu()
            scores.append(batch_scores)
            #print("batch scores appended")

    if scores:
        scores_tensor = torch.cat(scores, dim=0)
    else:
        scores_tensor = torch.empty(0, dtype=torch.float32)

    mean_similarity = float(scores_tensor.mean().item()) if scores_tensor.numel() > 0 else float('nan')
    return mean_similarity, scores_tensor.tolist()


def evaluate_polynomial(c, distance, angle):
    if len(c) != 6:
        print("Number of coefficients is wrong.")
    return c[0] + c[1]*distance + c[2]*angle + c[3]*distance**2 + c[4]*angle + c[5]*angle*distance