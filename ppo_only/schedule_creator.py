"""
schedule looks like
rpy_flag bound

rpy_flag values:
0 - roll
1 - pitch
2 - yaw

bound is in radians

python schedule_creator.py --default True --mode 0/1- using presets
python schedule_creator.py --default False --mode 0/1 - using user input
"""
import argparse

rpy_flag = 0
lower_bound = 0.0
upper_bound = 3.14
step_size = 0.1
bound_changes = 32

DEFAULT_C = True
DEFAULT_TYPE = 1
DEFAULT_MODE = 0
#Type = 0 is for incrementing, type = 1 is by dividing interval based on number
#of total changes given

def create(default=DEFAULT_C, division_type=DEFAULT_TYPE, mode=DEFAULT_MODE):
    global rpy_flag, lower_bound, upper_bound, step_size, bound_changes
    if(not(default)):
        rpy_flag = int(input("Enter 0 to set roll as the variable\nEnter 1 to\
                set pitch as the variable\nEnter 2 to set yaw as the variable\n"))
        lower_bound = float(input("Enter the lower bound value\n"))
        upper_bound = float(input("Enter the upper bound value\n"))
        step_size = float(input("Enter the step size\n"))
        bound_changes = int(input("Alternatively, enter the number of bound\
        changes\n"))
    with open("schedule.txt", "w") as fp:
        if(DEFAULT_TYPE == 0):
            i = lower_bound
            while(i <= upper_bound):
                fp.write(f"{rpy_flag} {i}\n")
                i += step_size
        else:
            bounds = [float(i)*((upper_bound-lower_bound)/bound_changes) for i in range(0, bound_changes+1)]
            if(mode == 0):
                for i in bounds:
                    fp.write(f"{rpy_flag} {i}\n")
            else:
                for i in range(len(bounds)-2, -1, -1):
                    fp.write(f"{rpy_flag} {bounds[i]} {upper_bound}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Schedule creator')
    parser.add_argument('--default', default=DEFAULT_C, type=bool,
                        help='False goes to input mode, True uses preset')
    parser.add_argument('--division_type', default=DEFAULT_TYPE, type=int,
            help='Sets how to create the bound schedule')
    parser.add_argument('--mode', default=DEFAULT_MODE, type=int,
            help='Sets which mode of sampling')
    ARGS = parser.parse_args()
    create(**vars(ARGS))
