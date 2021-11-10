import numpy as np
import pandas as pd
from utils import getNearestValue

foldername = "../data"
savepath_solution = foldername + "/volumes/solution_volume.csv"
savepath_tip = foldername + "/tips/tips.csv"

experimentpath = foldername + "/nextquery/csv/r11_data8min.csv"
sourcepath = foldername + "/sources/protocol_source8.csv"


FGFRipath = "./FGFRi_conc.csv"
KSRpath = "./KSR_conc.csv"

columns_pipets=["10mL","200mL","1000mL","5000mL"]
columns_solution = ["A液+FGFRi","10%KSR液","KSR原液","3因子"]

if __name__ == "__main__":
    
    experiment_df = pd.read_csv(experimentpath)
    source_df = pd.read_csv(sourcepath)
    FGFRi_df = pd.read_csv(FGFRipath, encoding="cp932")
    KSR_df = pd.read_csv(KSRpath)
    
    #solution volume
    datas_nparray = np.array([])
    for protocol_number in range(len(source_df)):
        dataframe = source_df[protocol_number:protocol_number+1]
        FGFRi_vol = 0.
        KSR10_vol = 0.
        KSR100_vol = 0.
        factorA3_vol = 0.
        factorB3_vol = 0.
        for i in range(len(list(source_df))-1):
            if (i%5 == 0):
                # FGFRi濃度集計
                FGFRi_conc = dataframe[str(i)].values[0]
                if (FGFRi_conc == "-") or (type(FGFRi_conc) == bool):
                    FGFRi_vol += 0.
                else:
                    FGFRi_conc = float(FGFRi_conc)
                    #print("FGFRi_conc ",FGFRi_conc) 
                    #print("volume", FGFRi_df.loc[FGFRi_df.conc == FGFRi_conc]["add_volume"].values[0])
                    FGFRi_vol +=  FGFRi_df.loc[FGFRi_df.conc == FGFRi_conc]["add_volume"].values[0]
                    #print("FGFRi_vol ",FGFRi_vol)

            elif (i%5 == 3):
                # KSR濃度集計
                KSR_conc = dataframe[str(i)].values[0]
                if (KSR_conc == "-") or (type(KSR_conc) == bool):
                    hoge=1
                else:
                    KSR_conc = float(KSR_conc)
                    KSR10_vol += KSR_df.loc[KSR_df.conc == KSR_conc]["volume10"].values[0]
                    KSR100_vol += KSR_df.loc[KSR_df.conc == KSR_conc]["volume原液"].values[0]

            elif (i%5 == 4):
                # 3因子on/off集計
                factor3_flag = str(dataframe[str(i)].values[0])
                if factor3_flag == "TRUE":
                    factorA3_vol +=5
                    factorB3_vol +=10
                else:
                    continue

        # プロトコルごとに["A液+FGFRi","10%KSR液","KSR原液","3因子"]まとめる
        #print(FGFRi_vol)
        array=np.array([FGFRi_vol/1000, KSR10_vol/1000, KSR100_vol/1000, factorA3_vol/1000,factorB3_vol/1000])
        datas_nparray = np.append(datas_nparray, array)

    # シートをセーブ
    datas_nparray = datas_nparray.reshape((len(source_df),5))
    df2 = pd.DataFrame(datas_nparray)
    df2.to_csv(savepath_solution)
    
    #tips
    datas_nparray = np.array([])
    print(len(source_df))
    for protocol_number in range(len(source_df)):
        dataframe = source_df[protocol_number:protocol_number+1]
        num_10uL = 0
        num_200uL = 0
        num_1000uL = 0
        num_5000uL = 0
        if protocol_number <12:
            num_10uL = 0
            num_200uL = 24
            num_1000uL = 4
            num_5000uL = 28
        elif 13<protocol_number:
            num_10uL = 0
            num_200uL = 24
            num_1000uL = 0
            num_5000uL = 24     
        else:
            num_10uL = 0
            num_200uL = 36
            num_1000uL = 32
            num_5000uL = 28             

        for i in range(len(list(source_df))-1):
            if (i%5 == 0):
                # FGFRi濃度集計
                FGFRi_conc = dataframe[str(i)].values[0]
                if (FGFRi_conc == "-") or (type(FGFRi_conc) == bool):
                    hoge=1
                else:
                    FGFRi_conc = float(FGFRi_conc)
                    #print("FGFRi_conc ",FGFRi_conc) 
                    #print("volume", FGFRi_df.loc[FGFRi_df.conc == FGFRi_conc]["add_volume"].values[0])
                    conc_info = FGFRi_df.loc[FGFRi_df.conc == FGFRi_conc]
                    num_10uL += conc_info["10"].values[0] + conc_info["5"].values[0]
                    num_200uL += conc_info["200"].values[0] + conc_info["80"].values[0] + conc_info["30"].values[0]
                    num_1000uL += conc_info["1000"].values[0] + conc_info["450"].values[0] + conc_info["300"].values[0]

            elif (i%5 == 3):
                # KSR濃度集計
                KSR_conc = dataframe[str(i)].values[0]
                if (KSR_conc == "-") or (type(KSR_conc) == bool):
                    hoge=1
                else:
                    KSR_conc = float(KSR_conc)
                    conc_info = KSR_df.loc[KSR_df.conc == KSR_conc]
                    num_10uL += conc_info["10"].values[0] + conc_info["5"].values[0] \
                                + conc_info["10.1"].values[0] + conc_info["5.1"].values[0]
                    num_200uL += conc_info["200"].values[0] + conc_info["80"].values[0] + conc_info["30"].values[0] \
                                + conc_info["200.1"].values[0] + conc_info["80.1"].values[0] + conc_info["30.1"].values[0]
                    num_1000uL += conc_info["1000"].values[0] + conc_info["450"].values[0] + conc_info["300"].values[0] \
                                + conc_info["1000.1"].values[0] + conc_info["450.1"].values[0] + conc_info["300.1"].values[0]
                    num_5000uL += conc_info["3000"].values[0] \
                                + conc_info["3000.1"].values[0]

            elif (i%5 == 4):
                # 3因子on/off集計
                factor3_flag = str(dataframe[str(i)].values[0])
                if factor3_flag == "TRUE":
                    num_10uL += 1 + 1
                else:
                    continue

        # プロトコルごとにチップ消費量まとめる
        array=np.array([num_5000uL, num_1000uL, num_200uL, num_10uL])
        datas_nparray = np.append(datas_nparray, array)

    # シートをセーブ
    print(datas_nparray.shape)
    datas_nparray = datas_nparray.reshape((len(source_df),4))
    df2 = pd.DataFrame(datas_nparray)
    df2.to_csv(savepath_tip)
    
