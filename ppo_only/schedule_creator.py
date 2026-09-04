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
python schedule_creator.py --division_type 0 - incrementing
python schedule_creator.py --division_type 1 - dividing interval based on
number of changes given
python schedule_creator.py --mode 0 - 3 - mode of changing intervals
"""
import argparse

pos_orn_flag = 0
flag = 2
xyz_lower_bound = 1.0
xyz_upper_bound = 10.0
rpy_lower_bound = 0.0
rpy_upper_bound = 3.14
xyz_step_size = 1.0
rpy_step_size = 0.1
xyz_bound_changes = 10
rpy_bound_changes = 32
reward_type = 1

DEFAULT_C = True
DEFAULT_DIVISION_TYPE = 1
DEFAULT_MODE = 1
def get_bounds(setting, lower_bound=0, upper_bound=3.14, step_size=0.1, bound_changes=32):
    vals = []
    if(setting == 0):
        i = lower_bound
        while(i <= upper_bound):
            vals.append(i)
            i += step_size
    else:
        vals = [(lower_bound + (float(i)*((upper_bound-lower_bound)/bound_changes))) for i in range(0, bound_changes+1)]
    return vals

def create(default=DEFAULT_C, division_type=DEFAULT_DIVISION_TYPE, mode=DEFAULT_MODE):
    global pos_orn_flag, flag, reward_type
    global xyz_lower_bound, xyz_upper_bound, xyz_step_size, xyz_bound_changes
    global rpy_lower_bound, rpy_upper_bound, rpy_step_size, rpy_bound_changes
    if(not(default)):
        pos_orn_flag = int(input("Enter 0 if you want to modify position\n\
                Enter 1 if you want to modify orientation\n"))
        if(not(pos_orn_flag)):
            flag = int(input("Enter 0 to set x as the variable\nEnter 1 to\
                    set y as the variable\nEnter 2 to set z as the variable\n"))
            xyz_lower_bound = float(input("Enter the lower bound value\n"))
            xyz_upper_bound = float(input("Enter the upper bound value\n"))
            xyz_step_size = float(input("Enter the step size\n"))
            xyz_bound_changes = int(input("Alternatively, enter the number of bound\
                changes\n"))
        else:
            _flag = int(input("Enter 0 to set roll as the variable\nEnter 1 to\
                    set pitch as the variable\nEnter 2 to set yaw as the variable\n"))
            rpy_lower_bound = float(input("Enter the lower bound value\n"))
            rpy_upper_bound = float(input("Enter the upper bound value\n"))
            rpy_step_size = float(input("Enter the step size\n"))
            rpy_bound_changes = int(input("Alternatively, enter the number of bound\
                changes\n"))
        reward_type = int(input("Enter 1 if you want sparse rewards\nEnter 0 if\
                you want dense rewards\n"))
    with open("schedule.txt", "w") as fp:
        if(not(pos_orn_flag)):
            bounds = get_bounds(division_type, xyz_lower_bound, xyz_upper_bound,
                    xyz_step_size, xyz_bound_changes)
        else:
            bounds = get_bounds(division_type, rpy_lower_bound, rpy_upper_bound,
                    rpy_step_size, rpy_bound_changes)
        write_choices = []
        if(mode == 0):
            for i in range(len(bounds)):
                write_choices.append([-bounds[i], bounds[i]])
        elif(mode == 1):
            for i in range(len(bounds)-2, -1, -1):
                if(pos_orn_flag):
                    write_choices.append([bounds[i], rpy_upper_bound])
                else:
                    write_choices.append([bounds[i], xyz_upper_bound])
        elif(mode == 2):
            for i in range(1, len(bounds)):
                write_choices.append([bounds[i-1], bounds[i]])
        elif(mode == 3):
            for i in range(len(bounds)-2, -1, -1):
                write_choices.append([bounds[i], bounds[i+1]])
        for i in write_choices:
            fp.write(f"{pos_orn_flag} {flag} {mode} {i[0]} {i[1]} {reward_type}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Schedule creator')
    parser.add_argument('--default', default=DEFAULT_C, type=bool,
                        help='False goes to input mode, True uses preset')
    parser.add_argument('--division_type', default=DEFAULT_DIVISION_TYPE, type=int,
            help='Sets how to create the bound schedule')
    parser.add_argument('--mode', default=DEFAULT_MODE, type=int,
            help='Sets which mode of sampling')
    ARGS = parser.parse_args()
    create(**vars(ARGS))
