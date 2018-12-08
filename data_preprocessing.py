import pandas as pd
import argparse
import os
import numpy as np

from torch.utils.data.dataset import Dataset

class SepsisDataset(Dataset):
    def __init__(self, data_path, transforms=None):
        self.before_path = os.path.join(data_path, "sepsis_before.csv")
        self.after_path = os.path.join(data_path, "sepsis_after.csv")
        self.before_data = pd.read_csv(self.before_path)
        self.after_data = pd.read_csv(self.after_path)
        self.transforms = transforms

    def __getitem__(self, index):
        x_before = np.array(self.before_data.iloc[index])
        x_after = np.array(self.after_data.iloc[index])
        if self.transforms is not None:
            x_before = self.transforms(x_before)
            x_after = self.transforms(x_after)
        return (x_before, x_after)

    def __len__(self):
        len_before = len(self.before_data)
        len_after = len(self.after_data)
        assert(len_before == len_after)
        return len_before

def read_data(file_loc, columns=None, chunksize=None):
    data = pd.read_csv(file_loc, delimiter=',', usecols=columns, chunksize=chunksize)
    return data

def main():
    parser = argparse.ArgumentParser(description='MIMIC III')
    parser.add_argument('--datadir', default='./data', type=str,
                        help='path to the directory that contains the data')
    parser.add_argument('--chunksize', default=1000000, type=int,
                        help='chunksize')
    parser.add_argument('--interpolate', default='linear', type=str,
                        help='how to interpolate missing values')
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
    SOFA - Sequential Organ Failure Assessment
    Age
    
    LAB VALUES
    Albumin
    Arterial pH
    Calcium
    Glucose
    Hemoglobin
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
    adm_cols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME']
    admissions_loc = os.path.join(args.datadir, "ADMISSIONS.csv")
    admissions = read_data(admissions_loc, columns=adm_cols)
    admissions.columns = admissions.columns.str.lower()

    """
    Sepsis Data
    """
    sepsis_loc = os.path.join(args.datadir, "sepsis3-df.csv")
    sepsis = read_data(sepsis_loc)

    joined_data = sepsis.merge(admissions, on=['hadm_id'])
    sepsis_patients = joined_data[['hadm_id', 'subject_id']]
    sepsis_patients_loc = os.path.join(args.datadir, "sepsis3-patients.csv")
    sepsis_patients.to_csv(sepsis_patients_loc, index=False)
    """
    D_ITEMS
    """
    d_items_loc = os.path.join(args.datadir, "D_ITEMS.csv")
    # d_items_cols = ['LABEL', 'ITEMID', 'DBSOURCE', 'LINKSTO', 'CATEGORY', 'UNITNAME', 'PARAM_TYPE', 'CONCEPTID']
    d_items_cols = ['LABEL', 'ITEMID']
    d_items = read_data(d_items_loc, columns=d_items_cols)
    d_items.columns = d_items.columns.str.lower()
    d_items.label = d_items.label.str.lower()

    d_items_select_loc = os.path.join(args.datadir, "D_ITEMS_select.csv")
    d_items_select = read_data(d_items_select_loc)
    d_items_select.label = d_items_select.label.str.lower()

    d_items_final = d_items_select.merge(d_items, on=['label'], how='left')
    d_items_final_loc = os.path.join(args.datadir, "D_ITEMS_final.csv")
    d_items_final.to_csv(d_items_final_loc, index=False)

    # """
    # CHARTEVENTS
    # """
    # chartevents_loc = os.path.join(args.datadir, "CHARTEVENTS.csv")
    # chartevents = pd.DataFrame()
    # for i, chunk in enumerate(read_data(chartevents_loc, chunksize=args.chunksize)):
    #     print("Merging chunk {} ...".format(i+1))
    #     chunk.columns = chunk.columns.str.lower()
    #     filtered_chunk = chunk.merge(d_items_final, on=['itemid'], how='inner')
    #     merged_chunk = pd.merge(sepsis_patients, filtered_chunk, on=['subject_id', 'hadm_id'], how='inner')
    #     chartevents = chartevents.append(merged_chunk)
    # chartevents_output = os.path.join(args.datadir, "sepsis_chartevents.csv")
    # chartevents.to_csv(chartevents_output, index=False)

    # """
    # CHARTEVENTS
    # """
    # chartevents_loc = os.path.join(args.datadir, "CHARTEVENTS.csv")
    # sofa_patients = pd.DataFrame()
    # for i, chunk in enumerate(read_data(chartevents_loc, chunksize=args.chunksize)):
    #     print("Merging chunk {} ...".format(i+1))
    #     chunk.columns = chunk.columns.str.lower()
    #     filtered_chunk = chunk.merge(d_items_final, on=['itemid'], how='inner')
    #     # merged_chunk = filtered_chunk[['subject_id', 'hadm_id']].drop_duplicates()
    #     filtered_chunk = filtered_chunk[filtered_chunk['itemid'] == 227428][['subject_id', 'hadm_id']].drop_duplicates()
    #     sofa_patients = sofa_patients.append(filtered_chunk)
    # sofa_patients_output = os.path.join(args.datadir, "sofa_patients.csv")
    # sofa_patients.to_csv(sofa_patients_output, index=False)

    """
    INPUTEVENTS_CV
    """
    input_events_cv_loc = os.path.join(args.datadir, "INPUTEVENTS_CV.csv")
    sofa_patients = pd.DataFrame()
    for i, chunk in enumerate(read_data(input_events_cv_loc, chunksize=args.chunksize)):
        print("Merging chunk {} ...".format(i+1))
        chunk.columns = chunk.columns.str.lower()
        filtered_chunk = chunk.merge(d_items_final, on=['itemid'], how='inner')
        filtered_chunk = filtered_chunk[filtered_chunk['itemid'] == 227428][['subject_id', 'hadm_id']].drop_duplicates()
        sofa_patients = sofa_patients.append(filtered_chunk)
    sofa_patients_output = os.path.join(args.datadir, "sofa_patients.csv")
    sofa_patients.to_csv(sofa_patients_output, index=False)


    # """
    # LABITEMS
    # """
    # labitems_loc = os.path.join(args.datadir, "D_LABITEMS.csv")
    # labitems = read_data(labitems_loc)

    # """
    # LABEVENTS.csv
    # """
    # labevents_loc = os.path.join(args.datadir, "LABEVENTS.csv")
    # merged = pd.DataFrame()
    # for i, labevents in enumerate(read_data(labevents_loc, chunksize=args.chunksize)):
    #     print("Merging chunk {} ...".format(i+1))
    #     labevents.columns = labevents.columns.str.lower()
    #     merged_chunk = pd.merge(joined_data, labevents, on=['subject_id', 'hadm_id'], how='inner')
    #     merged = merged.append(merged_chunk)
    #
    # # Join Sepsis and ADMISSIONS
    # output_loc = os.path.join(args.datadir, "joined_data.csv")
    # merged.to_csv(output_loc, index=False)
    # print("Saved joined data in {}".format(output_loc))

    """"
    Organize by Patient
    """
    # chartevents_columns = ['hadm_id', 'charttime', 'label', 'valuenum']
    # chartevents_loc = os.path.join(args.datadir, "sepsis_chartevents.csv")
    # chartevents = read_data(chartevents_loc, columns=chartevents_columns)
    # chartevents_wide = pd.pivot_table(chartevents, index=['hadm_id', 'charttime'], columns='label', values='valuenum')
    # chartevents_wide = chartevents_wide.reset_index(level=['hadm_id', 'charttime'])
    #
    # sepsis_chartevents = chartevents_wide.merge(joined_data, how='inner')
    # sepsis_chartevents['suspected_infection_time_poe'] = pd.to_datetime(sepsis_chartevents['suspected_infection_time_poe'])
    # sepsis_chartevents = sepsis_chartevents.dropna(subset=['suspected_infection_time_poe'])
    # sepsis_chartevents['charttime'] = pd.to_datetime(sepsis_chartevents['charttime'])
    # sepsis_chartevents['time_from_suspected_infection_time'] = sepsis_chartevents['suspected_infection_time_poe'] - sepsis_chartevents['charttime']
    # sepsis_chartevents['time_from_suspected_infection_time'] = sepsis_chartevents['time_from_suspected_infection_time'].dt.total_seconds().div(60).astype(int)
    # sepsis_chartevents['before_suspected_infection_time'] = sepsis_chartevents['time_from_suspected_infection_time'] > 0
    #
    # static_vars = ['hadm_id', 'age', 'is_male', 'race_white', 'race_black', 'race_hispanic', 'race_other',
    #                'metastatic_cancer', 'diabetes', 'height', 'weight', 'bmi', 'elixhauser_hospital',
    #                'before_suspected_infection_time']
    # sepsis_chartevents_static = sepsis_chartevents[static_vars].drop_duplicates()
    #
    # dynamic_vars = ['albumin', 'arterial blood pressure diastolic',
    #                 'arterial blood pressure systolic', 'arterial paco2', 'arterial pao2',
    #                 'arterial ph', 'bun', 'calcium', 'chloride', 'creatinine', 'glucose',
    #                 'heart rate', 'hemoglobin', 'inr', 'ionized calcium', 'lactic acid',
    #                 'magnesium', 'manual blood pressure diastolic left',
    #                 'manual blood pressure diastolic right',
    #                 'manual blood pressure systolic left',
    #                 'manual blood pressure systolic right',
    #                 'non invasive blood pressure diastolic',
    #                 'non invasive blood pressure systolic', 'platelets', 'potassium', 'pt',
    #                 'ptt', 'respiratory rate', 'sodium', 'temperature celsius',
    #                 'total bilirubin']
    #
    # group_vars = ['hadm_id', 'before_suspected_infection_time']
    # for var in dynamic_vars:
    #     aggregated = sepsis_chartevents.groupby(group_vars)[var].mean()
    #     aggregated = aggregated.reset_index(level=group_vars)
    #     sepsis_chartevents_static = sepsis_chartevents_static.merge(aggregated, on=group_vars)
    #
    # sepsis_chartevents_loc = os.path.join(args.datadir, "sepsis_aggregated.csv")
    # sepsis_chartevents_static.to_csv(sepsis_chartevents_loc, index=False)

    # # Interpolate missing values
    # sepsis_aggregated = read_data(sepsis_chartevents_loc)
    # sepsis_aggregated_before = sepsis_aggregated[sepsis_aggregated['before_suspected_infection_time'] == 1]
    # sepsis_aggregated_before_int = sepsis_aggregated_before.interpolate(method=args.interpolate, limit_direction='both')
    # sepsis_aggregated_after = sepsis_aggregated[sepsis_aggregated['before_suspected_infection_time'] == 0]
    # sepsis_aggregated_after_int = sepsis_aggregated_after.interpolate(method=args.interpolate, limit_direction='both')
    #
    # # Normalize continuous vars
    # dummy_vars = ['is_male', 'race_white', 'race_black', 'race_hispanic', 'race_other',
    #               'metastatic_cancer', 'diabetes']
    # vars_to_normalize = ['age', 'height', 'weight', 'bmi', 'elixhauser_hospital',
    #                      'albumin', 'arterial blood pressure diastolic',
    #                      'arterial blood pressure systolic', 'arterial paco2', 'arterial pao2',
    #                      'arterial ph', 'bun', 'calcium', 'chloride', 'creatinine', 'glucose',
    #                      'heart rate', 'hemoglobin', 'inr', 'ionized calcium', 'lactic acid',
    #                      'magnesium', 'manual blood pressure diastolic left',
    #                      'manual blood pressure diastolic right',
    #                      'manual blood pressure systolic left',
    #                      'manual blood pressure systolic right',
    #                      'non invasive blood pressure diastolic',
    #                      'non invasive blood pressure systolic', 'platelets', 'potassium', 'pt',
    #                      'ptt', 'respiratory rate', 'sodium', 'temperature celsius',
    #                      'total bilirubin']
    # sepsis_aggregated_before_int = sepsis_aggregated_before_int[dummy_vars + vars_to_normalize]
    # sepsis_aggregated_before_int[vars_to_normalize] = (sepsis_aggregated_before_int[vars_to_normalize] -
    #                                                    sepsis_aggregated_before_int[vars_to_normalize].mean()) / \
    #                                                    sepsis_aggregated_before_int[vars_to_normalize].std()
    # len_after = len(sepsis_aggregated_after_int)
    # len_before = len(sepsis_aggregated_before_int)
    # n_samples = len_after - len_before
    # samples = sepsis_aggregated_before_int.sample(n=n_samples, random_state=11, replace=True)
    # sepsis_aggregated_before_int = sepsis_aggregated_before_int.append(samples)
    # data_before_loc = os.path.join(args.datadir, "sepsis_before.csv")
    # sepsis_aggregated_before_int.to_csv(data_before_loc, index=False, header=False)
    #
    # sepsis_aggregated_after_int = sepsis_aggregated_after_int[dummy_vars + vars_to_normalize]
    # sepsis_aggregated_after_int[vars_to_normalize] = (sepsis_aggregated_after_int[vars_to_normalize] -
    #                                                   sepsis_aggregated_after_int[vars_to_normalize].mean()) / \
    #                                                   sepsis_aggregated_after_int[vars_to_normalize].std()
    # data_after_loc = os.path.join(args.datadir, "sepsis_after.csv")
    # sepsis_aggregated_after_int.to_csv(data_after_loc, index=False, header=False)

if __name__ == '__main__':
    main()
