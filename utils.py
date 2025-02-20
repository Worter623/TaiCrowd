import taichi as ti
import csv
from string import Template
import math
import pandas as pd

_max = int(0xBBBBBB)
_min = int(0x111111)
_range = _max - _min + 1

EPSILON = 0.00001
INF = math.inf

vec2 = ti.types.vector(2, float)
Line = ti.types.vector(4, float) #0,1:direction; 2,3:point

class FloatPair:
    """
    Defines a pair of scalar values.  NOT IN USE NOW 2021.3.31 (FOR KD-TREE)
    """

    def __init__(self, a, b):
        self.a_ = a
        self.b_ = b

    def __lt__(self, other):
        """
        Returns true if the first pair of scalar values is less than the second pair of scalar values.
        """
        return self.a_ < other.a_ or not (other.a_ < self.a_) and self.b_ < other.b_

    def __le__(self, other):
        """
        Returns true if the first pair of scalar values is less than or equal to the second pair of scalar values.
        """
        return (self.a_ == other.a_ and self.b_ == other.b_) or self < other

    def __gt__(self, other):
        """
        Returns true if the first pair of scalar values is greater than the second pair of scalar values.
        """
        return not (self <= other)

    def __ge__(self, other):
        """
        Returns true if the first pair of scalar values is greater than or equal to the second pair of scalar values.
        """
        return not (self < other)

# A dictionary of all the supported possible crowd simulation algorithms. 
# ORCA:0, social force:1, steering behavior:2
METHODS = {'orca': 0,'ORCA':0, 'social_force': 1, 'SOCIAL_FORCE':1,'steering': 2,'STEERING':2}

@ti.func
def convert_data(_data):
    """Map _data[0,1] to different hex color ranges"""
    hex_color = int(_data * _range + _min)
    return hex_color

def export_Astar(map,pixel_num,batch,filename):
    """export Astar result as csv, file path can be specified through param filename"""
    headers = ['pixel_ID', 'astarX', 'astarY']
    dtypes = {'pixel_ID': int, 'astarX': float, 'astarY': float}  # set dtype for each col
    df = pd.DataFrame(columns=headers)
    df = df.astype(dtypes)  # set dtype for DataFrame
    for i in range(pixel_num):
        result = map[batch,i]
        df.loc[i] = [int(i), result[0], result[1]]
    df.to_csv(filename, index=False)

def read_astar_from_csv(map,pixel_num,batch,filename):
    """set Astar result with data from csv. data format must follow the format in func export_Astar"""
    df = pd.read_csv(filename)
    if len(df) != pixel_num:
        raise ValueError("The number of rows in the CSV file does not match the expected length.")
    
    for i in range(pixel_num):
        vec = ti.Vector([df.at[i, 'astarX'],df.at[i, 'astarY']])
        map[batch,i] = vec
    return map

def export_pos0(N,pos,belong_batch,path):
    """export initial position and batch of agents as csv, file path can be specified through param path"""
    headers = ['ID', 'PosX', 'PosY', 'batch']
    dtypes = {'ID': int, 'PosX': float, 'PosY': float, 'batch':int}  # set dtype for each col
    df = pd.DataFrame(columns=headers)
    df = df.astype(dtypes)  # set dtype for DataFrame
    for i in range(N):
        df.loc[i] = [int(i), pos[i][0], pos[i][1],int(belong_batch[i])]
    df.to_csv(path, index=False)

def read_pos0_from_csv(pos,filename):
    """set initial position and batch with data from csv. data format must follow the format in func export_pos0"""
    df = pd.read_csv(filename)
    N = len(pos)
    if len(df) != N:
        raise ValueError("The number of rows in the CSV file does not match the expected length.")
    total_batch = int(df.at[N-1,'batch']+1)
    group = [0]*total_batch
    for i in range(N):
        pos[i][0] = df.at[i, 'PosX']
        pos[i][1] = df.at[i, 'PosY']
        belong_batch = int(df.at[i, 'batch'])
        group[belong_batch] += 1
    return pos,total_batch,group

def export_csv(data,path):
    """Write a csv file in the specified format. The path is specified by param path."""
    headers = ['---', 'CrowdPos','CrowdID']
    #(X=2383.000,Y=983.000,Z=0.000)
    template = Template('(X=${s1},Y=${s2},Z=0.00)')

    with open(path, "w", encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        i = 0
        for row in data:
            column = []
            num_frames = len(row)
            if num_frames <= 670:
                # If the total number of frames is less than or equal to the required total number of frames, no sampling is performed.
                frames_to_sample = row
            else:
                # Calculate the sampling interval to ensure that the final column contains the specified number of frames
                frame_skip = num_frames // 670
                frames_to_sample = row[::frame_skip]

            for pos in frames_to_sample:
                str_temp = template.safe_substitute(s1=float(f'{pos[0]:.2f}'), s2=float(f'{pos[1]:.2f}'))
                column.append(str_temp)
                
            # Combine the entire column into a single string, separated by commas
            columns = f"({','.join(column)})"
            label = "NewRow_"
            label += str(i)
            writer.writerow([label, columns,i])
            i += 1

# for UE
def export_csv_UE(data,path):
    """Write a csv file in the specified format. The path is specified by param path."""
    headers = ['---', 'Loc2D']
    template = Template('(Locs=(X=${s1},Y=${s2}))')

    with open(path, "w", encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        i = 0
        for row in data:
            column = []
            num_frames = len(row)
            if num_frames <= 670:
                # If the total number of frames is less than or equal to the required total number of frames, no sampling is performed.
                frames_to_sample = row
            else:
                # Calculate the sampling interval to ensure that the final column contains the specified number of frames
                frame_skip = num_frames // 670
                frames_to_sample = row[::frame_skip]

            for pos in frames_to_sample:
                str_temp = template.safe_substitute(s1=float(f'{pos[0]:.2f}'), s2=float(f'{pos[1]:.2f}'))
                column.append(str_temp)
                
            # Combine the entire column into a single string, separated by commas
            columns = f"({','.join(column)})"

            writer.writerow([i, columns])
            i += 1

# for Transformer
def export_txt_Transformer(data,path):
    """Write a txt file in the specified format. The path is specified by param path."""
    template = Template('${frame} ${id} Pedestrian -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ${xpos} -1.0 ${ypos} -1.0\n')
    with open(path, 'w') as file:
        i = 0
        for row in data:
            num_frames = len(row)
            if num_frames <= 670:
                # If the total number of frames is less than or equal to the required total number of frames, no sampling is performed.
                frames_to_sample = row
            else:
                # Calculate the sampling interval to ensure that the final column contains the specified number of frames
                frame_skip = num_frames // 670
                frames_to_sample = row[::frame_skip]

            j = 0
            for pos in frames_to_sample:
                str_temp = template.safe_substitute(frame=j, id=i,xpos=float(f'{pos[0]:.2f}'), ypos=float(f'{pos[1]:.2f}'))
                j += 1
                file.write(str_temp)
            i += 1

@ti.func
def normalize(vec):
    """Vector Normalization"""
    norm = vec.norm()
    new_vec = ti.Vector([0.0, 0.0])
    if norm != 0:
        new_vec = vec / norm
    return new_vec, norm

def _normalize(vec):
    """Vector Normalization"""
    norm = vec.norm()
    new_vec = ti.Vector([0.0, 0.0])
    if norm != 0:
        new_vec = vec / norm
    return new_vec, norm

@ti.func
def expand(vec,a,b):
    return ti.Vector([vec[0]*a,vec[1]*b])

@ti.func
def capped(vec, limit):
    """Scale down a desired velocity to its capped speed."""
    norm = vec.norm()
    new_vec = ti.Vector([0.0, 0.0])
    if norm != 0:
        new_vec = vec * ti.min(1, limit/norm)
    return new_vec


@ti.func
def vector_angles(vec):
    """Calculate angles for an array of vectors  = atan2(y, x)"""
    return ti.atan2(vec[1], vec[0])

@ti.func
def set_mag(v: ti.template(), mag: ti.f32):
    return (v / v.norm()) * mag


@ti.func
def limit(a, mag):
    norm = a.norm()
    return ti.select(norm > 0 and norm > mag, (a / norm) * mag, a)

@ti.func
def check_in_list(field:ti.template(),element:ti.i32,len:ti.i32)->ti.i8:
    """
    Check if an element exists in the field list. Return 1 if it exists.
    len: length you want to check
    """
    flag = 0
    for i in range (len):
        if flag == 1:
            continue
        if field[i] == element:
            flag = 1
    return flag

@ti.func
def abs_sq(vector1:vec2,vector2:vec2)->ti.f64:
    """
    input should be vec2
    """
    return vector1[0]*vector2[0] + vector1[1] * vector2[1]

@ti.func
def det(vector1:vec2,vector2:vec2)->ti.f64:
    return vector1[0] * vector2[1] - vector1[1] * vector2[0]

def _det(vector1,vector2):
    return vector1[0] * vector2[1] - vector1[1] * vector2[0]

def left_of(a, b, c):
    """
    Computes the signed distance from a line connecting the specified points to a specified point.

    Args:
        a (Vector2): The first point on the line.
        b (Vector2): The second point on the line.
        c (Vector2): The point to which the signed distance is to be calculated.

    Returns:
        float: Positive when the point c lies to the left of the line ab.
    """
    return _det(a - c, b - a)

@ti.func
def value_limit(value,minv,maxv):
    """limit value in [min,max]"""
    v = ti.min(value,maxv)
    v = ti.max(value,minv)
    return v

