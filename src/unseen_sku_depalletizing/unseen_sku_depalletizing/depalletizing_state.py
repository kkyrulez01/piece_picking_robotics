from enum import Enum, auto

class ExecutionMode(Enum):
    MANUAL = auto()
    AUTO = auto()

class DepalletizingPhase(Enum):
    IDLE = auto()

    PICK = auto()
    PLACE = auto()
    COMPLETE = auto()

    ERROR = auto()

class PickState(Enum):
    WAITING_FOR_PICK_POINT = auto()

    WAITING_FOR_OBSTACLE_CLOUD = auto()
    WAITING_FOR_OCTOMAP = auto()

    READY_FOR_PRE_PICK = auto()
    PLANNING_PRE_PICK = auto()
    EXECUTING_PRE_PICK = auto()

    PLANNING_PICK = auto()
    EXECUTING_PICK = auto()

    ACTIVATING_SUCTION = auto()

    PLANNING_RETRACT = auto()
    EXECUTING_RETRACT = auto()

    PICK_FINISHED = auto()

class PlaceState(Enum):
    READY_FOR_PRE_PLACE = auto()

    PLANNING_PRE_PLACE = auto()
    EXECUTING_PRE_PLACE = auto()

    PLANNING_PLACE = auto()
    EXECUTING_PLACE = auto()

    RELEASING_SUCTION = auto()

    PLANNING_RETRACT = auto()
    EXECUTING_RETRACT = auto()

    PLACE_FINISHED = auto()

class CompleteState(Enum):
    CHECKING_FOR_MORE_OBJECTS = auto()

    PLANNING_HOME = auto()
    EXECUTING_HOME = auto()

    COMPLETE_FINISHED = auto()