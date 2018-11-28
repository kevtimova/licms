import pandas as pd
import argparse
import os

def read_admissions(file_loc, columns=None):
    admissions = pd.read_csv(file_loc, delimiter=',', usecols=columns)
    import ipdb; ipdb.set_trace()

def main():
    parser = argparse.ArgumentParser(description='MIMIC III')
    parser.add_argument('--datadir', default='./data', type=str,
                        help='path to the directory that contains the data')
    # Get arguments
    args = parser.parse_args()

    # Read data
    """
    ADMISSIONS.csv
    
    PATIENTS.csv
    SUBJECT_ID, GENDER, DOB, DOD, DOD_HOSP, DOD_SSN, EXPIRE_FLAG
    """
    admissions_loc = os.path.join(args.datadir, "ADMISSIONS.csv")
    read_admissions(admissions_loc)


if __name__ == '__main__':
    main()
