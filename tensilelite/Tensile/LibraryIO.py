################################################################################
#
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from .Common import printExit, printWarning, versionIsCompatible
from .CustomKernels import getCustomKernelConfig
from .SolutionStructs import Solution, ProblemSizes, ProblemType
from . import Parallel
from . import __version__
from . import Common
from . import SolutionLibrary
from .CustomYamlLoader import load_yaml_stream

from typing import NamedTuple, List
import os
import sys

import multiprocessing
from multiprocessing import cpu_count
from multiprocessing import Manager, Pool
from itertools import repeat
from copy import deepcopy
import math

manager = Manager()

try:
    import orjson as json
except ImportError:
    try:
        import ujson as json
        printWarning("orjson not installed. Fallback to ujson.")
    except ImportError:
        try:
            import simplejson as json
            printWarning("orjson, ujson not installed. Fallback to simplejson.")
        except ImportError:
            import json
            printWarning("orjson, ujson, simplejson not installed. Fallback to json.")

try:
    import yaml
except ImportError:
    printExit(
        "You must install PyYAML to use Tensile (to parse config files). See http://pyyaml.org/wiki/PyYAML for installation instructions."
    )

try:
    from yaml import CSafeLoader as yamlLoader
except ImportError:
    from yaml import SafeLoader as yamlLoader
    printWarning("CSafeLoader not installed. Fallback to SafeLoader.")

try:
    import msgpack
except ImportError:
    print("Message pack python library not detected. Must use YAML backend instead.")


###################
# Writing functions
###################
def write(filename_noExt, data, format="yaml"):
    """Writes data to file with specified format; extension is appended based on format."""
    if format == "yaml":
        writeYAML(filename_noExt + ".yaml", data)
    elif format == "json":
        writeJson(filename_noExt + ".json", data)
    elif format == "msgpack":
        writeMsgPack(filename_noExt + ".dat", data)
    else:
        printExit("Unrecognized write format {}".format(format))


def writeYAML(filename, data, **kwargs):
    """Writes data to file in YAML format."""
    # set default kwags for yaml dump
    if "explicit_start" not in kwargs:
        kwargs["explicit_start"] = True
    if "explicit_end" not in kwargs:
        kwargs["explicit_end"] = True
    if "default_flow_style" not in kwargs:
        kwargs["default_flow_style"] = None

    with open(filename, "w") as f:
        yaml.dump(data, f, **kwargs)

def writeJson(filename, data):
    """Writes data to file in json format."""
    with open(filename, "w") as f:
        json_object = json.dumps(data, option=json.OPT_INDENT_2).decode("utf-8") if 'orjson' in sys.modules else json.dumps(data, indent=2)
        f.write(json_object)

def writeMsgPack(filename, data):
    """Writes data to file in Message Pack format."""
    with open(filename, "wb") as f:
        msgpack.pack(data, f)

def writeSolutions(filename, problemSizes, biasTypeArgs, activationArgs, solutions, cache=False):
    """Writes solution YAML file."""

    # convert objects to nested dictionaries
    solutionStates = []

    if cache:
        solYaml = read(filename)
        if biasTypeArgs and activationArgs:
            solutionStates = solYaml[4:]
        elif biasTypeArgs or activationArgs:
            solutionStates = solYaml[3:]
        else:
            solutionStates = solYaml[2:]
    else:
        for solution in solutions:
            solutionState = solution.getAttributes()
            solutionState["ProblemType"] = solutionState["ProblemType"].state
            solutionState["ProblemType"]["DataType"] = \
                    solutionState["ProblemType"]["DataType"].value
            solutionState["ProblemType"]["DataTypeA"] = \
                    solutionState["ProblemType"]["DataTypeA"].value
            solutionState["ProblemType"]["DataTypeB"] = \
                    solutionState["ProblemType"]["DataTypeB"].value
            solutionState["ProblemType"]["DataTypeE"] = \
                    solutionState["ProblemType"]["DataTypeE"].value
            solutionState["ProblemType"]["DataTypeAmaxD"] = \
                    solutionState["ProblemType"]["DataTypeAmaxD"].value
            solutionState["ProblemType"]["DestDataType"] = \
                    solutionState["ProblemType"]["DestDataType"].value
            solutionState["ProblemType"]["ComputeDataType"] = \
                    solutionState["ProblemType"]["ComputeDataType"].value
            solutionState["ProblemType"]["BiasDataTypeList"] = \
                    [btype.value for btype in solutionState["ProblemType"]["BiasDataTypeList"]]
            solutionState["ProblemType"]["ActivationComputeDataType"] = \
                    solutionState["ProblemType"]["ActivationComputeDataType"].value
            solutionState["ProblemType"]["ActivationType"] = \
                    solutionState["ProblemType"]["ActivationType"].value
            solutionState["ProblemType"]["F32XdlMathOp"] = \
                solutionState["ProblemType"]["F32XdlMathOp"].value
            if "DataTypeMetadata" in solutionState["ProblemType"]:
                solutionState["ProblemType"]["DataTypeMetadata"] = \
                    solutionState["ProblemType"]["DataTypeMetadata"].value
            solutionStates.append(solutionState)
    # write dictionaries
    with open(filename, "w") as f:
        f.write("- MinimumRequiredVersion: {}\n".format(__version__))
        f.write("- ProblemSizes:\n")
        if problemSizes:
            for sizeRange in problemSizes.ranges:
                f.write("  - Range: {}\n".format(sizeRange))
            for problemExact in problemSizes.exacts:
                #FIXME-problem, this ignores strides:
                f.write("  - Exact: {}\n".format(problemExact))
        if biasTypeArgs:
            f.write("- BiasTypeArgs: [{}]\n".format([btype.value for btype in biasTypeArgs.biasTypes]))
        if activationArgs:
            f.write("- ActivationArgs:\n")
            for setting in activationArgs.settingList:
                f.write("  - [Enum: %s]\n"%(setting.activationEnum))
        yaml.dump(solutionStates, f, default_flow_style=None)


###############################
# Reading and parsing functions
###############################
def read(filename, customizedLoader=False):
    name, extension = os.path.splitext(filename)
    if extension == ".yaml":
        return load_yaml_stream(filename, yamlLoader) if customizedLoader else readYAML(filename)
    if extension == ".json":
        return readJson(filename)
    else:
        printExit("Unrecognized read format {}".format(extension))

def readYAML(filename):
    """Reads and returns YAML data from file."""
    with open(filename, "r") as f:
        data = yaml.load(f, yamlLoader)
    return data

def readJson(filename):
    """Reads and returns JSON data from file."""
    with open(filename, "r") as f:
        data = json.loads(f.read())
    return data

def parseSolutionsFile(filename):
    """Wrapper function to read and parse a solutions file."""
    return parseSolutionsData(read(filename), filename)


def parseSolutionsData(data, srcFile="?"):
    """Parses problem sizes and solutions from the data of a solutions file."""
    if len(data) < 3:
        printExit("Solution file {} is missing required fields (len = {} < 3" \
                .format(srcFile, len(data)))

    versionString = data[0]["MinimumRequiredVersion"]
    if not versionIsCompatible(versionString):
        printWarning("Version = {} in solution file {} does not match Tensile version = {}" \
                .format(srcFile, versionString, __version__) )

    if "ProblemSizes" not in data[1]:
        printExit("Solution file {} doesn't begin with ProblemSizes".format(srcFile))

    problemSizesConfig = data[1]["ProblemSizes"]
    solutionStartIdxInData = 2
    if (len(data) > solutionStartIdxInData) and "BiasTypeArgs" in data[solutionStartIdxInData]:
        solutionStartIdxInData += 1
    if (len(data) > solutionStartIdxInData) and "ActivationArgs" in data[solutionStartIdxInData]:
        solutionStartIdxInData += 1

    solutions = []
    for i in range(solutionStartIdxInData, len(data)):
        solutionState = data[i]
        # force redo the deriving of parameters, make sure old version logic yamls can be validated
        solutionState["AssignedProblemIndependentDerivedParameters"] = False
        solutionState["AssignedDerivedParameters"] = False
        solutionObject = Solution(solutionState)
        solutions.append(solutionObject)
    problemType = solutions[0]["ProblemType"]
    problemSizes = ProblemSizes(problemType, problemSizesConfig)
    return (problemSizes, solutions)


class LibraryLogic(NamedTuple):
    """Return tuple for parseLibraryLogicData()"""
    schedule: str
    architecture: str
    problemType: ProblemType
    solutions: list
    exactLogic: list
    library: SolutionLibrary.MasterSolutionLibrary
    srcFile: str

import time

def parseLibraryLogicFile(filename, archs=None):
    f = read(filename, True)
    """Wrapper function to read and parse a library logic file."""
    return parseLibraryLogicData(f, filename, archs)

def parseLibraryLogicDataAndFilter(data, srcFile="?", archs=None):
    """Parses the data of a library logic file."""
    if isinstance(data, List):
        data = parseLibraryLogicList(data, srcFile)

    is_arch_valid = lambda cArch, tArch : (cArch == tArch or cArch == "all")
    if not (archs is None) and "ArchitectureName" in data:
        if isinstance(archs, List):
            if len(archs) > 0 and not archs[0] == "all":
                if not (any(is_arch_valid(arch.split(":")[0], data["ArchitectureName"]) for arch in archs)):
                    return dict()
        elif isinstance(archs, str):
            if not is_arch_valid(archs.split(":")[0], data["ArchitectureName"]):
                return dict()

    if "CUCount" not in data:
        data["CUCount"] = None

    if not versionIsCompatible(data["MinimumRequiredVersion"]):
        printWarning("Version = {} in library logic file {} does not match Tensile version = {}" \
                .format(srcFile, data["MinimumRequiredVersion"], __version__) )

    return data


def parUnpackSoln(solutionList, solutions, tid, n_cores, data):
    npt = math.ceil(len(solutionList) / n_cores)

    # unpack solution
    def solutionStateToSolution(solutionState, data) -> Solution:

        # If parameter not in yaml, fill with default values
        if "KernelLanguage" not in solutionState.keys():
            solutionState["KernelLanguage"] = Common.defaultSolution["KernelLanguage"]
        if "CustomKernelName" not in solutionState.keys():
            solutionState["CustomKernelName"] = Common.defaultSolution["CustomKernelName"]

        solutionState["ProblemType"] = data["ProblemType"]

        if solutionState["KernelLanguage"] == "Assembly":
            solutionState["ISA"] = Common.gfxArch(data["ArchitectureName"])
        else:
            solutionState["ISA"] = (0, 0, 0)
        solutionState["CUCount"] = data["CUCount"]
        # force redo the deriving of parameters, make sure old version logic yamls can be validated
        solutionState["AssignedProblemIndependentDerivedParameters"] = False
        solutionState["AssignedDerivedParameters"] = False
        if solutionState["CustomKernelName"]:
            isp = {}
            if "InternalSupportParams" in solutionState:
                isp = solutionState["InternalSupportParams"]
            customConfig = getCustomKernelConfig(solutionState["CustomKernelName"], isp)
            for key, value in customConfig.items():
                solutionState[key] = value
        solutionObject = Solution(solutionState)

        return solutionObject

    res = []
    for i in range(npt * tid, min(npt * (tid + 1), len(solutionList))):
        solutionState = solutionList[i]
        res.append(solutionStateToSolution(solutionState, data))
    solutions.extend(res)

        
def parseLibraryLogicData(data, srcFile="?", archs=None):
    """Parses the data of a library logic file."""
    if isinstance(data, List):
        data = parseLibraryLogicList(data, srcFile)

    is_arch_valid = lambda cArch, tArch : (cArch == tArch or cArch == "all")
    if not (archs is None) and "ArchitectureName" in data:
        if isinstance(archs, List):
            if len(archs) > 0 and not archs[0] == "all":
                if not (any(is_arch_valid(arch.split(":")[0], data["ArchitectureName"]) for arch in archs)):
                    return LibraryLogic("", "", None, [], [], None, srcFile)
        elif isinstance(archs, str):
            if not is_arch_valid(archs.split(":")[0], data["ArchitectureName"]):
                return LibraryLogic("", "", None, [], [], None, srcFile)

    if "CUCount" not in data:
        data["CUCount"] = None

    if not versionIsCompatible(data["MinimumRequiredVersion"]):
        printWarning("Version = {} in library logic file {} does not match Tensile version = {}" \
                .format(srcFile, data["MinimumRequiredVersion"], __version__) )

    # unpack problemType
    problemType = ProblemType(data["ProblemType"])

    #solutions = [solutionStateToSolution(solutionState) for solutionState in data["Solutions"]]

    dataLite = {}
    dataLite["CUCount"] = data["CUCount"]
    dataLite["ArchitectureName"] = data["ArchitectureName"]
    dataLite["ProblemType"] = data["ProblemType"]
    
    n_cores = min( int(len(data["Solutions"]) / 650), Parallel.CPUThreadCount())
    #print("len soln", len(data["Solutions"]))
    solutions = None
    #t0 = time.time()
    if n_cores <= 1:
        solutions = []
        parUnpackSoln(data["Solutions"], solutions, 0, 1, dataLite)
    else:
        solutions = manager.list()
        tid = range(0, n_cores)

        with Pool(n_cores) as p:
            p.starmap(parUnpackSoln, zip(repeat(data["Solutions"]), repeat(solutions), tid, repeat(n_cores), repeat(dataLite)))
    #t1 = time.time()
    #print("Time to read all files:", t1 - t0)
    
    del data["Solutions"]    
    newLibrary, _ = SolutionLibrary.MasterSolutionLibrary.FromOriginalState(data, solutions)
    del data["ProblemType"]

    rv = LibraryLogic(data["ScheduleName"], data["ArchitectureName"], problemType, solutions, \
             data.get("ExactLogic"), newLibrary, srcFile)
    return rv

def parseLibraryLogicList(data, srcFile="?"):
    """Parses the data of a matching table style library logic file."""
    if len(data) < 9:
        printExit("Library logic file {} is missing required fields (len = {} < 9)" \
                .format(srcFile, len(data)))

    rv = {}
    rv["MinimumRequiredVersion"] = data[0]["MinimumRequiredVersion"]
    rv["ScheduleName"] = data[1]
    rv["DeviceNames"] = data[3]
    rv["ProblemType"] = data[4]
    rv["Solutions"] = data[5]

    if type(data[2]) is dict:
        rv["ArchitectureName"] = data[2]["Architecture"]
        rv["CUCount"] = data[2]["CUCount"]
    else:
        rv["ArchitectureName"] = data[2]
        rv["CUCount"] = None

    # TODOBEN: figure out what to do with these...
    rv["ExactLogic"] = data[7]
    rv["RangeLogic"] = data[8]

    # optional fields
    if len(data) > 10 and data[10]:
        rv["PerfMetric"] = data[10]

    # library logic fields
    libraryType = None
    if len(data) > 11 and data[11]:
        libraryType = data[11]
    else:
        printExit("Library logic file {} is missing required field matching property." \
                .format(srcFile))
    if libraryType == "FreeSize":
        rv["LibraryType"] = "FreeSize"
        rv["Library"] = {}
        rv["Library"]["indexOrder"] = None
        rv["Library"]["table"] = [0, len(data[5])]
        rv["Library"]["distance"] = None
    else:
        rv["LibraryType"] = "Matching"
        rv["Library"] = {}
        rv["Library"]["indexOrder"] = data[6]
        rv["Library"]["table"] = data[7]
        rv["Library"]["distance"] = libraryType

    return rv

def rawLibraryLogic(data):
    """Returns a tuple of the data in a library logic file."""
    versionString = data[0]
    scheduleName = data[1]
    architectureName = data[2]
    deviceNames = data[3]
    problemTypeState = data[4]
    solutionStates = data[5]
    indexOrder = data[6]
    exactLogic = data[7]
    rangeLogic = data[8]
    otherFields = []

    dataLength = len(data)
    if dataLength > 9:
        for idx in range(9, dataLength):
            otherFields.append(data[idx])

    return (versionString, scheduleName, architectureName, deviceNames,\
            problemTypeState, solutionStates, indexOrder, exactLogic, rangeLogic, otherFields)

#################
# Other functions
#################
def createLibraryLogic(schedulePrefix, architectureName, deviceNames, libraryType, logicTuple):
    """Creates the data for a library logic file suitable for writing to YAML."""
    problemType = logicTuple[0]
    solutions = logicTuple[1]
    indexOrder = logicTuple[2]
    exactLogic = logicTuple[3]
    rangeLogic = logicTuple[4]

    tileSelection = False
    if len(logicTuple) > 5 and logicTuple[5]:
        tileSelection = True

    data = []
    # Tensile version
    data.append({"MinimumRequiredVersion": __version__})
    # schedule name
    data.append(schedulePrefix)  # change from Tensile to vega10
    data.append(architectureName)
    # schedule device names
    data.append(deviceNames)
    # problem type
    problemTypeState = problemType.state
    problemTypeState["DataType"] = \
            problemTypeState["DataType"].value
    problemTypeState["DataTypeA"] = \
            problemTypeState["DataTypeA"].value
    problemTypeState["DataTypeB"] = \
            problemTypeState["DataTypeB"].value
    problemTypeState["DataTypeE"] = \
            problemTypeState["DataTypeE"].value
    problemTypeState["DataTypeAmaxD"] = \
            problemTypeState["DataTypeAmaxD"].value
    problemTypeState["DestDataType"] = \
            problemTypeState["DestDataType"].value
    problemTypeState["ComputeDataType"] = \
            problemTypeState["ComputeDataType"].value
    problemTypeState["BiasDataTypeList"] = \
            [btype.value for btype in problemTypeState["BiasDataTypeList"]]
    problemTypeState["ActivationComputeDataType"] = \
            problemTypeState["ActivationComputeDataType"].value
    problemTypeState["ActivationType"] = \
            problemTypeState["ActivationType"].value
    problemTypeState["F32XdlMathOp"] = \
            problemTypeState["F32XdlMathOp"].value
    if "DataTypeMetadata" in problemTypeState:
        problemTypeState["DataTypeMetadata"] = \
                problemTypeState["DataTypeMetadata"].value
    data.append(problemTypeState)

    # remove parameters with are set to the default values
    # so they are copied to the yaml files
    def removeDefaultVals(params):
        for k in list(params.keys()):
            if k in Common.defaultSolution.keys():
                if params[k] == Common.defaultSolution[k]:
                    del params[k]

    # solutions
    solutionList = []
    for solution in solutions:
        solutionState = solution.getAttributes()
        removeDefaultVals(solutionState)
        if "ProblemType" in solutionState.keys():
            del solutionState["ProblemType"]
        solutionList.append(solutionState)

    if tileSelection:
        tileSolutions = logicTuple[5]
        for solution in tileSolutions:
            solutionState = solution.getAttributes()
            removeDefaultVals(solutionState)
            if "ProblemType" in solutionState.keys():
                del solutionState["ProblemType"]
            solutionList.append(solutionState)

    data.append(solutionList)
    # index order
    data.append(indexOrder)

    # exactLogic
    exactLogicList = []
    if exactLogic:
        for key in exactLogic:
            exactLogicList.append([list(key), exactLogic[key]])
        data.append(exactLogicList)
    else:
        data.append(None)

    # rangeLogic
    data.append(rangeLogic)

    if tileSelection:
        tileSelectionLogic = {}
        tileSelectionIndices = logicTuple[6]
        tileSelectionLogic["TileSelectionIndices"] = tileSelectionIndices
        data.append(tileSelectionLogic)
    else:
        data.append(None)

    data.append(logicTuple[7]) # PerfMetric
    data.append(libraryType) # LibraryType
    return data
