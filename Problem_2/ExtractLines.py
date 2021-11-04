#!/usr/bin/env python3

############################################################
# ExtractLines.py
#
# This script reads in range data from a csv file, and
# implements a split-and-merge to extract meaningful lines
# in the environment.
############################################################

# Imports
import numpy as np
from PlotFunctions import *


############################################################
# functions
############################################################

def ExtractLines(RangeData, params):
    '''
    This function implements a split-and-merge line extraction algorithm.

    Inputs:
        RangeData: (x_r, y_r, theta, rho)
            x_r: robot's x position (m).
            y_r: robot's y position (m).
            theta: (1D) np array of angle 'theta' from data (rads).
            rho: (1D) np array of distance 'rho' from data (m).
        params: dictionary of parameters for line extraction.
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        segend: np array (N_lines, 4) of line segment endpoints. Each row represents [x1, y1, x2, y2].
        pointIdx: (N_lines,2) segment's first and last point index.
    '''

    # Extract useful variables from RangeData
    x_r = RangeData[0]
    y_r = RangeData[1]
    theta = RangeData[2]
    rho = RangeData[3]

    ### Split Lines ###
    N_pts = len(rho)
    r = np.zeros(0)
    alpha = np.zeros(0)
    pointIdx = np.zeros((0, 2), dtype=np.int)

    # This implementation pre-prepartitions the data according to the "MAX_P2P_DIST"
    # parameter. It forces line segmentation at sufficiently large range jumps.
    rho_diff = np.abs(rho[1:] - rho[:(len(rho)-1)])
    LineBreak = np.hstack((np.where(rho_diff > params['MAX_P2P_DIST'])[0]+1, N_pts))
    startIdx = 0
    for endIdx in LineBreak:
        alpha_seg, r_seg, pointIdx_seg = SplitLinesRecursive(theta, rho, startIdx, endIdx, params)
        N_lines = r_seg.size

        ### Merge Lines ###
        if (N_lines > 1):
            alpha_seg, r_seg, pointIdx_seg = MergeColinearNeigbors(theta, rho, alpha_seg, r_seg, pointIdx_seg, params)
        r = np.append(r, r_seg)
        alpha = np.append(alpha, alpha_seg)
        pointIdx = np.vstack((pointIdx, pointIdx_seg))
        startIdx = endIdx

    N_lines = alpha.size

    ### Compute endpoints/lengths of the segments ###
    segend = np.zeros((N_lines, 4))
    seglen = np.zeros(N_lines)
    for i in range(N_lines):
        rho1 = r[i]/np.cos(theta[pointIdx[i, 0]]-alpha[i])
        rho2 = r[i]/np.cos(theta[pointIdx[i, 1]-1]-alpha[i])
        x1 = rho1*np.cos(theta[pointIdx[i, 0]])
        y1 = rho1*np.sin(theta[pointIdx[i, 0]])
        x2 = rho2*np.cos(theta[pointIdx[i, 1]-1])
        y2 = rho2*np.sin(theta[pointIdx[i, 1]-1])
        segend[i, :] = np.hstack((x1, y1, x2, y2))
        seglen[i] = np.linalg.norm(segend[i, 0:2] - segend[i, 2:4])

    ### Filter Lines ###
    # Find and remove line segments that are too short
    goodSegIdx = np.where((seglen >= params['MIN_SEG_LENGTH']) &
                          (pointIdx[:, 1] - pointIdx[:, 0] >= params['MIN_POINTS_PER_SEGMENT']))[0]
    pointIdx = pointIdx[goodSegIdx, :]
    alpha = alpha[goodSegIdx]
    r = r[goodSegIdx]
    segend = segend[goodSegIdx, :]

    # change back to scene coordinates
    segend[:, (0, 2)] = segend[:, (0, 2)] + x_r
    segend[:, (1, 3)] = segend[:, (1, 3)] + y_r

    return alpha, r, segend, pointIdx


def SplitLinesRecursive(theta, rho, startIdx, endIdx, params):
    '''
    This function executes a recursive line-slitting algorithm, which
    recursively sub-divides line segments until no further splitting is
    required.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        startIdx: starting index of segment to be split.
        endIdx: ending index of segment to be split.
        params: dictionary of parameters.
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        idx: (N_lines,2) segment's first and last point index.

    HINT: Call FitLine() to fit individual line segments.
    HINT: Call FindSplit() to find an index to split at.
    '''
    ########## Code starts here ##########
    alpha, r = FitLine(theta[startIdx: endIdx], rho[startIdx: endIdx])
    if endIdx - startIdx <= params['MIN_POINTS_PER_SEGMENT']:
        # not enough points in the list to split
        return alpha, r, np.array([[startIdx, endIdx]])
    splitAtIdx = FindSplit(theta[startIdx: endIdx], rho[startIdx: endIdx], alpha, r, params)
    if splitAtIdx == -1:
        # all points within the vicinity of line (alpha, r)
        return alpha, r, np.array([[startIdx, endIdx]])
    else:
        # split the list around startIdx + splitAtIdx
        alpha_1, r_1, idx_1 = SplitLinesRecursive(theta, rho, startIdx, startIdx + splitAtIdx, params)
        alpha_2, r_2, idx_2 = SplitLinesRecursive(theta, rho, startIdx + splitAtIdx, endIdx, params)
        alpha = np.append(alpha_1, alpha_2)
        r = np.append(r_1, r_2)
        idx = np.vstack((idx_1, idx_2))
    ########## Code ends here ##########
    return alpha, r, idx

def FindSplit(theta, rho, alpha, r, params):
    '''
    This function takes in a line segment and outputs the best index at which to
    split the segment, or -1 if no split should be made.

    The best point to split at is the one whose distance from the line is
    the farthest, as long as this maximum distance exceeds
    LINE_POINT_DIST_THRESHOLD and also does not divide the line into segments
    smaller than MIN_POINTS_PER_SEGMENT. Otherwise, no split should be made.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: 'alpha' of input line segment (1 number).
        r: 'r' of input line segment (1 number).
        params: dictionary of parameters.
    Output:
        splitIdx: idx at which to split line (return -1 if it cannot be split).

    Hint: the output is the index of the first element in the second line. 
    For instance, if theta = np.array([0,1,2,3,4,5,6]) and splitIdx = 3, 
    then the two line segments will be 
        theta1 = theta[0:splitIdx] = np.array([0,1,2]) and 
        theta2 = theta[splitIdx:] = np.array([3,4,5,6]).
    '''
    ########## Code starts here ##########
    _no_split = -1
    min_nodes_per_segment = params['MIN_POINTS_PER_SEGMENT']
    if len(theta) < 2 * min_nodes_per_segment:
        # not enough points to split into 2 segments
        return _no_split
    # range for eligible P
    p_range = np.arange(min_nodes_per_segment, len(theta) - min_nodes_per_segment + 1)
    node_distance = np.abs(rho[p_range] * np.cos(theta[p_range] - alpha) - r)
    if max(node_distance) <= params['LINE_POINT_DIST_THRESHOLD']:
        # only need to check distance of eligible P
        return _no_split
    else:
        return np.argmax(node_distance) + min_nodes_per_segment
    ########## Code ends here ##########
    # return splitIdx

def FitLine(theta, rho):
    '''
    This function outputs a least squares best fit line to a segment of range
    data, expressed in polar form (alpha, r).

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
    Outputs:
        alpha: 'alpha' of best fit for range data (1 number) (rads). Should be between -pi and pi.
        r: 'r' of best fit for range data (1 number) (m). Should be positive.
    '''
    ########## Code starts here ##########
    n_nodes = len(theta)
    # assume all weights w_i = 1
    num_ = np.sum(np.square(rho) * np.sin(2 * theta)) \
           - 2.0 / n_nodes * np.sum(np.outer(rho * np.cos(theta), rho * np.sin(theta)))
    den_ = np.sum(np.square(rho) * np.cos(2 * theta)) \
           - 1.0 / n_nodes * (np.sum(np.outer(rho * np.cos(theta), rho * np.cos(theta)))
                              - np.sum(np.outer(rho * np.sin(theta), rho * np.sin(theta))))
    alpha = .5 * np.arctan2(num_, den_) + np.pi / 2
    r = 1.0 / n_nodes * np.sum(rho * np.cos(theta - alpha))
    ########## Code ends here ##########
    return alpha, r

def MergeColinearNeigbors(theta, rho, alpha, r, pointIdx, params):
    '''
    This function merges neighboring segments that are colinear and outputs a
    new set of line segments.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        pointIdx: (N_lines,2) segment's first and last point indices.
        params: dictionary of parameters.
    Outputs:
        alphaOut: output 'alpha' of merged lines (rads).
        rOut: output 'r' of merged lines (m).
        pointIdxOut: output start and end indices of merged line segments.

    HINT: loop through line segments and try to fit a line to data points from
          two adjacent segments. If this line cannot be split, then accept the
          merge. If it can be split, do not merge.
    '''
    ########## Code starts here ##########
    line_i = 0
    while line_i < len(alpha) - 1:
        # attempt to merge line_i and line_i_plus_1
        startIdx = pointIdx[line_i][0]
        endIdx = pointIdx[line_i + 1][1]
        alpha_new_line, r_new_line = FitLine(theta[startIdx: endIdx], rho[startIdx: endIdx])
        split_new_line = FindSplit(theta[startIdx: endIdx], rho[startIdx: endIdx], alpha_new_line, r_new_line, params)
        if split_new_line != -1:
            # cannot merge 2 lines
            line_i += 1
        else:
            # two line segments are co-linear and should be merged
            alpha[line_i] = alpha_new_line
            r[line_i] = r_new_line
            pointIdx[line_i][1] = endIdx
            alpha = np.delete(alpha, line_i + 1)
            r = np.delete(r, line_i + 1)
            pointIdx = np.delete(pointIdx, line_i + 1, 0)
    ########## Code ends here ##########
    return alpha, r, pointIdx


#----------------------------------
# ImportRangeData
def ImportRangeData(filename):

    data = np.genfromtxt('./RangeData/'+filename, delimiter=',')
    x_r = data[0, 0]
    y_r = data[0, 1]
    theta = data[1:, 0]
    rho = data[1:, 1]
    return (x_r, y_r, theta, rho)
#----------------------------------


############################################################
# Main
############################################################
def main():
    # parameters for line extraction (mess with these!)
    MIN_SEG_LENGTH = 0.1  # minimum length of each line segment (m)
    LINE_POINT_DIST_THRESHOLD = 0.01  # max distance of pt from line to split
    MIN_POINTS_PER_SEGMENT = 2  # minimum number of points per line segment
    MAX_P2P_DIST = 1.0  # max distance between two adjacent pts within a segment

    # Data files are formated as 'rangeData_<x_r>_<y_r>_N_pts.csv
    # where x_r is the robot's x position
    #       y_r is the robot's y position
    #       N_pts is the number of beams (e.g. 180 -> beams are 2deg apart)

    # filename = 'rangeData_5_5_180.csv'
    # filename = 'rangeData_4_9_360.csv'
    filename = 'rangeData_7_2_90.csv'

    # Import Range Data
    RangeData = ImportRangeData(filename)

    params = {'MIN_SEG_LENGTH': MIN_SEG_LENGTH,
              'LINE_POINT_DIST_THRESHOLD': LINE_POINT_DIST_THRESHOLD,
              'MIN_POINTS_PER_SEGMENT': MIN_POINTS_PER_SEGMENT,
              'MAX_P2P_DIST': MAX_P2P_DIST}

    alpha, r, segend, pointIdx = ExtractLines(RangeData, params)

    ax = PlotScene()
    ax = PlotData(RangeData, ax)
    ax = PlotRays(RangeData, ax)
    ax = PlotLines(segend, ax)

    plt.show()

############################################################

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
