def us(spectra,
       group_by = 'ID'):

    '''
    Compute the uncertainty score(us) of experimental replicates of identical samples. 
    
    Parameters:
    -----
    spectra : {pandas.DataFrame}
        Raw or preprocessed spectra
    
    group_by : str
        the column in `spectra` that indicates the experimental replicates
        
        
    
    Return
    ------
    us : {pandas.Series}
        the uncertainty score of each sample

    norm_us : {pandas.Series}
        the normalized uncertainty score of each sample
    '''

    def _check_type(spectra):
        
        if not isinstance(spectra, pd.DataFrame):
            raise TypeError(f" 'spectra' should be a 'pandas.DataFrame' ")
            
        return spectra

    spectra = _check_type(spectra)

    assert group_by in spectra.columns, "Invalid 'group_by' "

    us = spectra.groupby(group_by).std().mean(axis=1)

    return (us, us/us.sum())



def csv_to_excel(csv_files,
                 excel_file_name,
                 excel_file_path=None):
    '''
    Save multiple csv files to an excel file
    
    Parameters:
    -----
    csv_files : list
        a list of csv files, e.g. ['file1.csv','file2.csv']
    
    excel_file_name : str
        the name of the excel file, e.g. "myfile.xlsx"
        
    
    excel_file_path: str
        the path to save the excel file, if None, the file will be save in the current directory
    
    Return
    ------
    None
    '''
    
    if not isinstance(csv_files, list):
        raise TypeError("'csv_files' must be a 'list' object!")
        
    if not isinstance(excel_file_name, str):
        raise TypeError("'excel_file_name' must be a 'str' object!")
    
    # Dictionary to store dataframes
    dfs = {}

    for file in csv_files:

        # Assuming the CSV files have the same structure

        df = pd.read_csv(file)

        # Extract the filename without extension as the sheet name

        sheet_name = os.path.basename(file).replace(".csv", "")

        dfs[sheet_name] = df
    
    # Create an Excel writer object
    if excel_file_path is not None:
        writer = pd.ExcelWriter(excel_file_path, engine='xlsxwriter')
    else:
        writer = pd.ExcelWriter(excel_file_name, engine='xlsxwriter')

    # Write each dataframe to a separate sheet in the Excel file
    for sheet_name, df in dfs.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Close the Excel writer
    writer.close()