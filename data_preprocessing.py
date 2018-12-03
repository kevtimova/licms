import pandas as pd
import argparse
import os

def read_data(file_loc, columns=None, chunksize=None):
    data = pd.read_csv(file_loc, delimiter=',', usecols=columns, chunksize=chunksize)
    return data

def main():
    parser = argparse.ArgumentParser(description='MIMIC III')
    parser.add_argument('--datadir', default='./data', type=str,
                        help='path to the directory that contains the data')
    parser.add_argument('--chunksize', default=1000000, type=int,
                        help='chunksize')
    # Get arguments
    args = parser.parse_args()

    # Read data
    """
    ADMISSIONS.csv
    
    PATIENTS.csv
    SUBJECT_ID, GENDER, DOB, DOD, DOD_HOSP, DOD_SSN, EXPIRE_FLAG
    
    LABEVENTS.csv
    
    D_LABITEMS.csv
    
    
    DEMOGRAPHIC/STATIC
    Shock Index
    Elixhauser
    SIRS
    Gender
    Re-admission
    GCS - Glasgow Coma Scale
    SOFA - Se- quential Organ Failure Assessment
    Age
    
    LAB VALUES
    Albumin
    Arterial pH
    Calcium
    Glucose
    Haemoglobin
    Magnesium
    PTT - Partial Thromboplastin Time
    Potassium
    SGPT - Serum Glutamic-Pyruvic Transaminase
    Arterial Blood Gas
    BUN - Blood Urea Nitrogen
    Chloride
    Bicarbonate
    INR - International Normalized Ratio
    Sodium
    Arterial Lactate
    CO2
    Creatinine
    Ionised Calcium
    PT - Prothrombin Time
    Platelets Count
    SGOT - Serum Glutamic-Oxaloacetic Transaminase
    Total bilirubin
    White Blood Cell Count
    
    VITAL SIGNS
    Diastolic Blood Pressure
    Systolic Blood Pressure
    Mean Blood Pressure
    PaCO2
    PaO2
    FiO2
    PaO/FiO2 ratio
    Respiratory Rate
    Temperature (Celsius)
    Weight (kg)
    Heart Rate
    SpO2
    
    INTAKE AND OUTPUT EVENTS
    Fluid Output - 4 hourly period
    Total Fluid Output
    Mechanical Ventilation    
    """

    # Admissions
    cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime',
            'has_chartevents_data']
    admissions_loc = os.path.join(args.datadir, "ADMISSIONS.csv")
    admissions = read_data(admissions_loc)
    admissions.columns = admissions.columns.str.lower()


    """
    Sepsis Data
    """
    sepsis_loc = os.path.join(args.datadir, "sepsis3-df.csv")
    sepsis = read_data(sepsis_loc)

    joined_data = sepsis.merge(admissions, on=['hadm_id'])
    sepsis_patients = joined_data[['hadm_id', 'subject_id']]

    """
    CHARTEVENTS
    """
    chartevents_loc = os.path.join(args.datadir, "CHARTEVENTS.csv")
    chartevents = pd.DataFrame()
    for i, chunk in enumerate(read_data(chartevents_loc, chunksize=args.chunksize)):
        print("Merging chunk {} ...".format(i+1))
        chunk.columns = chunk.columns.str.lower()
        merged_chunk = pd.merge(sepsis_patients, chunk, on=['subject_id', 'hadm_id'], how='inner')
        chartevents.append(merged_chunk)
        import ipdb; ipdb.set_trace()


    # """
    # LABEVENTS.csv
    # """
    # labevents_loc = os.path.join(args.datadir, "LABEVENTS.csv")
    # merged = pd.DataFrame()
    # for i, labevents in enumerate(read_data(labevents_loc, chunksize=args.chunksize)):
    #     print("Merging chunk {} ...".format(i+1))
    #     labevents.columns = labevents.columns.str.lower()
    #     merged_chunk = pd.merge(joined_data, labevents, on=['subject_id', 'hadm_id'], how='inner')
    #     merged.append(merged_chunk)
    #
    # # Join Sepsis and ADMISSIONS
    # output_loc = os.path.join(args.datadir, "joined_data.csv")
    # merged.to_csv(output_loc)
    # print("Saved joined data in {}".format(output_loc))

if __name__ == '__main__':
    main()
