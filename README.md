This code is used for the reconstruction of unpacked data derived from the EBeTe unpacker used in ECAL Test Beam analysis

Overview of the code:
  - **reco.py & reco_functions.py**\
    python script for the reconstruction, where all the useful variables for DQM plots are saved in a tree + plotting (...)

  - **plot_functions_....py**\
    python script for plotting, very general with one single function for the different plots

  - **plot_list.csv**\
    csv file with the list of the plots and the settings for each one

for tests:

python3 reco.py -i /eos/cms/store/group/dpg_ecal/comm_ecal/upgrade/testbeam/ECALTB_H4_Oct2025/DataTree/19435/0001.root -r 19435 -s 1 -ro /tmp/prova_ruben -opt electrons -po /eos/user/r/rgargiul/www/test_ferrari

