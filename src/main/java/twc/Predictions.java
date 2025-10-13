package twc;

import environment.Utilities;
import optimizationTasks.ModelMixer;
import org.bson.Document;
import predictorTasks.BacktestingProcessor;

import java.util.ArrayList;
import java.util.List;

public class Predictions {


    public static void main(String[] args) {

        String KPI = "C:/_db/kpis/";
        String OUT = "C:/_arcturus/2025-10-13/";

        for (String[] model : MODELS_1024orig /*MODELS_64vega1*/) {
            String tickerId = model[0];
            int buyThreshold = Integer.parseInt(model[1]);
            int sellThreshold = Integer.parseInt(model[2]);
            String[] config = model[3].split(",");
            int multiplier = Integer.parseInt(model[4]);
            int neurons = Integer.parseInt(model[5]);
            String[] buyModels = model[6].split(",");
            String[] sellModels = model[7].split(",");
            String[] periods = {"2024", "2025"};
            backtestModels(tickerId, KPI, OUT, buyModels, sellModels, buyThreshold, sellThreshold, periods, config, multiplier, neurons);
            //break;
        }
    }


    private static void backtestModels(String tickerId, String KPIS, String OUT, String[] buyModels, String[] sellModels, int buyThreshold, int sellThreshold, String[] periods, String[] config, int multiplier, int neurons) {
            String kpiFile = KPIS + tickerId + "_kpis.txt";
            String outFile = OUT + String.format("%s_%s_forecast.txt", tickerId, Utilities.getTimeTag() );
            StringBuilder sb1 = new StringBuilder();

            BacktestingProcessor bp = new BacktestingProcessor(tickerId, String.format("backtest for %s", tickerId));
            bp.loadDataSet(kpiFile, periods, config, multiplier);
            List<String> modelFiles = new ArrayList<>();
            char[] modelVector = new char[buyModels.length+sellModels.length];
            int i = 0;
            for( ; i<buyModels.length; i++ ){
                modelFiles.add(buyModels[i]);
                modelVector[i]='B';
            }
            for( int j=0; j<sellModels.length; j++ ){
                modelFiles.add(sellModels[j]);
                modelVector[i++]='S';
            }
            ModelMixer mx = ModelMixer.createFromModelFiles(modelFiles,neurons,bp);
            mx.recalculateSignals(modelVector, buyThreshold, sellThreshold);
            Document result = bp.backtesting(mx.signals[0], mx.signals[1]);

            sb1.append( result.toJson() ).append("\n\n--------------------------------------------\n");
            System.out.println(result.toJson());
            bp.writeTradingHistoryDetail(sb1,mx);
            sb1.append("\n--------------------------------------------\n");
            mx.writeForecastDetailMatrix(bp,sb1);
            Utilities.writeFile(outFile, sb1);

    }



    final static String TRAINING_VECTOR = "cmf,obv,willR,atrPct,kcMPct,kcUPct,macdv,macdvSignal";
    final static String[][] MODELS_64vega1 = {
            {"AAPL", "1", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/GS_network2.txt,C:/_db/nets64_vega1/KO_network2.txt",
                    "C:/_db/nets64_vega1/COIN_network1.txt,C:/_db/nets64_vega1/CRM_network2.txt,C:/_db/nets64_vega1/SIL_network2.txt"},
            {"AMGN", "1", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/DIS_network2.txt,C:/_db/nets64_vega1/MCD_network2.txt,C:/_db/nets64_vega1/V_network2.txt",
                    "C:/_db/nets64_vega1/MRK_network2.txt"},
            {"AMZN", "1", "2", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/MPW_network1.txt",
                    "C:/_db/nets64_vega1/HD_network1.txt,C:/_db/nets64_vega1/IBM_network1.txt,C:/_db/nets64_vega1/KO_network1.txt,C:/_db/nets64_vega1/SIL_network1.txt"
                            + ",C:/_db/nets64_vega1/SIL_network2.txt,C:/_db/nets64_vega1/TRV_network1.txt"},
            {"AXP", "1", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/CSCO_network2.txt,C:/_db/nets64_vega1/IBM_network2.txt,C:/_db/nets64_vega1/MCD_network2.txt," +
                            "C:/_db/nets64_vega1/MPW_network2.txt,C:/_db/nets64_vega1/MRK_network2.txt,C:/_db/nets64_vega1/NKE_network1.txt," +
                            "C:/_db/nets64_vega1/NVDA_network1.txt,C:/_db/nets64_vega1/SIL_network2.txt,C:/_db/nets64_vega1/UNH_network1.txt," +
                            "C:/_db/nets64_vega1/VZ_network2.txt",
                    "C:/_db/nets64_vega1/AMZN_network2.txt,C:/_db/nets64_vega1/AXP_network2.txt,C:/_db/nets64_vega1/BA_network1.txt,C:/_db/nets64_vega1/COIN_network1.txt" +
                            ",C:/_db/nets64_vega1/COIN_network2.txt,C:/_db/nets64_vega1/GDXJ_network2.txt,C:/_db/nets64_vega1/JNJ_network2.txt,C:/_db/nets64_vega1/PLUG_network1.txt" +
                            ",C:/_db/nets64_vega1/SIL_network1.txt"},
            {"BA", "4", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/AMZN_network1.txt,C:/_db/nets64_vega1/CAT_network2.txt,C:/_db/nets64_vega1/COIN_network1.txt,C:/_db/nets64_vega1/CSCO_network1.txt" +
                            ",C:/_db/nets64_vega1/NVDA_network1.txt,C:/_db/nets64_vega1/TRV_network1.txt,C:/_db/nets64_vega1/VZ_network1.txt",
                    "C:/_db/nets64_vega1/BA_network1.txt,C:/_db/nets64_vega1/GDXJ_network1.txt,C:/_db/nets64_vega1/MCD_network2.txt" +
                            ",C:/_db/nets64_vega1/MPW_network1.txt,C:/_db/nets64_vega1/WMT_network2.txt"},
            {"CAT", "2", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/CSCO_network2.txt,C:/_db/nets64_vega1/CVX_network1.txt,C:/_db/nets64_vega1/JNJ_network1.txt" +
                            ",C:/_db/nets64_vega1/V_network1.txt,C:/_db/nets64_vega1/WMT_network2.txt",
                    "C:/_db/nets64_vega1/AMZN_network1.txt,C:/_db/nets64_vega1/BA_network1.txt,C:/_db/nets64_vega1/GS_network1.txt" +
                            ",C:/_db/nets64_vega1/JPM_network1.txt,C:/_db/nets64_vega1/MCD_network2.txt,C:/_db/nets64_vega1/MPW_network1.txt" +
                            ",C:/_db/nets64_vega1/MSFT_network1.txt,C:/_db/nets64_vega1/SHW_network1.txt,C:/_db/nets64_vega1/SIL_network1.txt"},
            {"CRM", "4", "3", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/AMZN_network1.txt,C:/_db/nets64_vega1/CSCO_network1.txt,C:/_db/nets64_vega1/CVX_network1.txt" +
                            ",C:/_db/nets64_vega1/MCD_network1.txt,C:/_db/nets64_vega1/NKE_network1.txt,C:/_db/nets64_vega1/SIL_network1.txt",
                    "C:/_db/nets64_vega1/BA_network2.txt,C:/_db/nets64_vega1/HD_network1.txt,C:/_db/nets64_vega1/MRK_network1.txt" +
                            ",C:/_db/nets64_vega1/MSFT_network2.txt,C:/_db/nets64_vega1/NNBR_network2.txt,C:/_db/nets64_vega1/NVDA_network2.txt" +
                            ",C:/_db/nets64_vega1/SHW_network2.txt,C:/_db/nets64_vega1/V_network1.txt"},
            {"CSCO", "1", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/COIN_network1.txt,C:/_db/nets64_vega1/COIN_network2.txt,C:/_db/nets64_vega1/MSFT_network2.txt" +
                            ",C:/_db/nets64_vega1/PG_network2.txt,C:/_db/nets64_vega1/TRV_network2.txt",
                    "C:/_db/nets64_vega1/AMZN_network2.txt,C:/_db/nets64_vega1/AXP_network2.txt,C:/_db/nets64_vega1/NKE_network2.txt" +
                            ",C:/_db/nets64_vega1/NVDA_network2.txt,C:/_db/nets64_vega1/SHW_network2.txt"},
            {"CVX", "1", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/CAT_network2.txt,C:/_db/nets64_vega1/HD_network1.txt,C:/_db/nets64_vega1/JPM_network1.txt,C:/_db/nets64_vega1/KO_network2.txt"
                            + ",C:/_db/nets64_vega1/MCD_network2.txt,C:/_db/nets64_vega1/MMM_network2.txt,C:/_db/nets64_vega1/MPW_network1.txt,C:/_db/nets64_vega1/NKE_network1.txt"
                            + ",C:/_db/nets64_vega1/PLUG_network2.txt,C:/_db/nets64_vega1/SHW_network1.txt,C:/_db/nets64_vega1/WMT_network2.txt",
                    "C:/_db/nets64_vega1/CAT_network2.txt,C:/_db/nets64_vega1/HD_network1.txt,C:/_db/nets64_vega1/JPM_network1.txt"},
            {"DIS", "1", "2", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/AAPL_network2.txt,C:/_db/nets64_vega1/AXP_network2.txt,C:/_db/nets64_vega1/GDXJ_network2.txt" +
                            ",C:/_db/nets64_vega1/PLUG_network1.txt,C:/_db/nets64_vega1/WMT_network2.txt",
                    "C:/_db/nets64_vega1/AMZN_network2.txt,C:/_db/nets64_vega1/DIS_network2.txt,C:/_db/nets64_vega1/PG_network2.txt" +
                            ",C:/_db/nets64_vega1/UNH_network2.txt"},
            {"GS", "2", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/CRM_network1.txt,C:/_db/nets64_vega1/JPM_network1.txt,C:/_db/nets64_vega1/SIL_network1.txt,C:/_db/nets64_vega1/V_network1.txt",
                    "C:/_db/nets64_vega1/GDXJ_network1.txt,C:/_db/nets64_vega1/MPW_network1.txt,C:/_db/nets64_vega1/NKE_network1.txt,C:/_db/nets64_vega1/NVDA_network2.txt"},
            {"HD", "1", "2", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/CRM_network2.txt,C:/_db/nets64_vega1/HD_network1.txt,C:/_db/nets64_vega1/MCD_network2.txt,C:/_db/nets64_vega1/PG_network1.txt",
                    "C:/_db/nets64_vega1/CAT_network2.txt,C:/_db/nets64_vega1/CRM_network1.txt,C:/_db/nets64_vega1/GDXJ_network1.txt,C:/_db/nets64_vega1/GS_network1.txt" +
                            ",C:/_db/nets64_vega1/HD_network2.txt,C:/_db/nets64_vega1/MSFT_network2.txt,C:/_db/nets64_vega1/NKE_network1.txt,C:/_db/nets64_vega1/NNBR_network1.txt" +
                            ",C:/_db/nets64_vega1/NVDA_network2.txt"},
            {"HON", "2", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/AAPL_network1.txt,C:/_db/nets64_vega1/COIN_network2.txt,C:/_db/nets64_vega1/CRM_network1.txt,C:/_db/nets64_vega1/CSCO_network1.txt" +
                            ",C:/_db/nets64_vega1/CSCO_network2.txt,C:/_db/nets64_vega1/IBM_network2.txt,C:/_db/nets64_vega1/MCD_network1.txt" +
                            ",C:/_db/nets64_vega1/MPW_network1.txt,C:/_db/nets64_vega1/TRV_network1.txt,C:/_db/nets64_vega1/UNH_network1.txt",
                    "C:/_db/nets64_vega1/AXP_network1.txt,C:/_db/nets64_vega1/BA_network1.txt,C:/_db/nets64_vega1/DIS_network1.txt" +
                            ",C:/_db/nets64_vega1/GS_network1.txt,C:/_db/nets64_vega1/MSFT_network2.txt,C:/_db/nets64_vega1/SHW_network1.txt" +
                            ",C:/_db/nets64_vega1/V_network2.txt"},
            {"IBM", "3", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/COIN_network2.txt,C:/_db/nets64_vega1/CRM_network2.txt,C:/_db/nets64_vega1/CVX_network2.txt" +
                            ",C:/_db/nets64_vega1/GDXJ_network2.txt,C:/_db/nets64_vega1/HON_network1.txt,C:/_db/nets64_vega1/IBM_network2.txt" +
                            ",C:/_db/nets64_vega1/JPM_network2.txt,C:/_db/nets64_vega1/MMM_network2.txt,C:/_db/nets64_vega1/MPW_network1.txt" +
                            ",C:/_db/nets64_vega1/MSFT_network2.txt,C:/_db/nets64_vega1/PG_network1.txt,C:/_db/nets64_vega1/PG_network2.txt" +
                            ",C:/_db/nets64_vega1/SHW_network2.txt,C:/_db/nets64_vega1/V_network2.txt,C:/_db/nets64_vega1/WMT_network2.txt",
                    "C:/_db/nets64_vega1/COIN_network1.txt,C:/_db/nets64_vega1/DIS_network2.txt,C:/_db/nets64_vega1/NNBR_network1.txt,C:/_db/nets64_vega1/NVDA_network2.txt"},
            {"JNJ", "2", "2", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/AAPL_network1.txt,C:/_db/nets64_vega1/AMGN_network1.txt,C:/_db/nets64_vega1/AXP_network1.txt" +
                            ",C:/_db/nets64_vega1/BA_network1.txt,C:/_db/nets64_vega1/VZ_network1.txt",
                    "C:/_db/nets64_vega1/AMZN_network2.txt,C:/_db/nets64_vega1/GDXJ_network1.txt,C:/_db/nets64_vega1/HON_network2.txt,C:/_db/nets64_vega1/KO_network2.txt" +
                            ",C:/_db/nets64_vega1/MCD_network1.txt,C:/_db/nets64_vega1/PG_network1.txt"},
            {"JPM", "1", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/AMZN_network2.txt,C:/_db/nets64_vega1/CAT_network2.txt,C:/_db/nets64_vega1/DIS_network1.txt,C:/_db/nets64_vega1/DIS_network2.txt" +
                            ",C:/_db/nets64_vega1/JNJ_network2.txt,C:/_db/nets64_vega1/JPM_network1.txt,C:/_db/nets64_vega1/MRK_network2.txt",
                    "C:/_db/nets64_vega1/JPM_network2.txt,C:/_db/nets64_vega1/MCD_network2.txt,C:/_db/nets64_vega1/NVDA_network1.txt"},
            {"KO", "1", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/JPM_network2.txt,C:/_db/nets64_vega1/SHW_network2.txt,C:/_db/nets64_vega1/SIL_network2.txt,C:/_db/nets64_vega1/V_network1.txt" +
                            ",C:/_db/nets64_vega1/V_network2.txt",
                    "C:/_db/nets64_vega1/AXP_network2.txt,C:/_db/nets64_vega1/MRK_network2.txt,C:/_db/nets64_vega1/MSFT_network2.txt"},
            {"MCD", "1", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/NKE_network1.txt,C:/_db/nets64_vega1/NNBR_network1.txt,C:/_db/nets64_vega1/NVDA_network1.txt,C:/_db/nets64_vega1/TRV_network1.txt",
                    "C:/_db/nets64_vega1/CRM_network2.txt,C:/_db/nets64_vega1/GS_network1.txt,C:/_db/nets64_vega1/MPW_network1.txt,C:/_db/nets64_vega1/UNH_network1.txt"},
            {"MMM", "1", "2", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/AAPL_network2.txt,C:/_db/nets64_vega1/COIN_network2.txt,C:/_db/nets64_vega1/DIS_network2.txt,C:/_db/nets64_vega1/SIL_network1.txt" +
                            ",C:/_db/nets64_vega1/WMT_network2.txt",
                    "C:/_db/nets64_vega1/AMGN_network2.txt,C:/_db/nets64_vega1/CVX_network1.txt,C:/_db/nets64_vega1/HD_network1.txt,C:/_db/nets64_vega1/JNJ_network2.txt" +
                            ",C:/_db/nets64_vega1/JPM_network2.txt,C:/_db/nets64_vega1/MPW_network2.txt,C:/_db/nets64_vega1/MRK_network2.txt" +
                            ",C:/_db/nets64_vega1/NKE_network1.txt,C:/_db/nets64_vega1/TRV_network2.txt"},
            {"MRK", "1", "3", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/CRM_network2.txt,C:/_db/nets64_vega1/HD_network1.txt,C:/_db/nets64_vega1/MMM_network1.txt,C:/_db/nets64_vega1/NNBR_network1.txt" +
                            ",C:/_db/nets64_vega1/SHW_network1.txt",
                    "C:/_db/nets64_vega1/COIN_network2.txt,C:/_db/nets64_vega1/HD_network2.txt,C:/_db/nets64_vega1/MCD_network2.txt,C:/_db/nets64_vega1/NKE_network2.txt" +
                            ",C:/_db/nets64_vega1/NNBR_network2.txt,C:/_db/nets64_vega1/TRV_network2.txt"},
            {"MSFT", "1", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/AMZN_network1.txt,C:/_db/nets64_vega1/CVX_network1.txt,C:/_db/nets64_vega1/MSFT_network1.txt",
                    "C:/_db/nets64_vega1/AMGN_network1.txt,C:/_db/nets64_vega1/GDXJ_network1.txt,C:/_db/nets64_vega1/MMM_network1.txt"},
            {"NKE", "1", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/AMGN_network2.txt,C:/_db/nets64_vega1/AXP_network1.txt,C:/_db/nets64_vega1/COIN_network1.txt,C:/_db/nets64_vega1/CVX_network2.txt" +
                            ",C:/_db/nets64_vega1/DIS_network1.txt,C:/_db/nets64_vega1/GS_network2.txt,C:/_db/nets64_vega1/HD_network1.txt,C:/_db/nets64_vega1/IBM_network1.txt" +
                            ",C:/_db/nets64_vega1/NVDA_network2.txt,C:/_db/nets64_vega1/VZ_network1.txt,C:/_db/nets64_vega1/WMT_network2.txt",
                    "C:/_db/nets64_vega1/AMZN_network2.txt,C:/_db/nets64_vega1/AXP_network2.txt,C:/_db/nets64_vega1/BA_network1.txt,C:/_db/nets64_vega1/CSCO_network2.txt" +
                            ",C:/_db/nets64_vega1/GDXJ_network1.txt,C:/_db/nets64_vega1/KO_network2.txt,C:/_db/nets64_vega1/MRK_network2.txt,C:/_db/nets64_vega1/SHW_network2.txt" +
                            ",C:/_db/nets64_vega1/TRV_network2.txt,C:/_db/nets64_vega1/V_network2.txt,C:/_db/nets64_vega1/WMT_network1.txt"},
            {"NVDA", "3", "2", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/AMGN_network2.txt,C:/_db/nets64_vega1/AXP_network1.txt,C:/_db/nets64_vega1/CAT_network1.txt,C:/_db/nets64_vega1/CRM_network1.txt" +
                            ",C:/_db/nets64_vega1/GS_network1.txt,C:/_db/nets64_vega1/TRV_network2.txt",
                    "C:/_db/nets64_vega1/AMZN_network1.txt,C:/_db/nets64_vega1/HD_network1.txt,C:/_db/nets64_vega1/MCD_network1.txt,C:/_db/nets64_vega1/MPW_network1.txt" +
                            ",C:/_db/nets64_vega1/PLUG_network1.txt"},
            {"PG", "1", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/AAPL_network1.txt,C:/_db/nets64_vega1/GS_network2.txt,C:/_db/nets64_vega1/JPM_network1.txt,C:/_db/nets64_vega1/JPM_network2.txt" +
                            ",C:/_db/nets64_vega1/NNBR_network2.txt,C:/_db/nets64_vega1/UNH_network1.txt",
                    "C:/_db/nets64_vega1/CRM_network1.txt,C:/_db/nets64_vega1/MCD_network2.txt,C:/_db/nets64_vega1/NKE_network1.txt,C:/_db/nets64_vega1/NKE_network2.txt" +
                            ",C:/_db/nets64_vega1/PG_network2.txt,C:/_db/nets64_vega1/SHW_network1.txt"},
            {"SHW", "1", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/AMGN_network1.txt,C:/_db/nets64_vega1/AMZN_network2.txt,C:/_db/nets64_vega1/JPM_network1.txt,C:/_db/nets64_vega1/MCD_network2.txt" +
                            ",C:/_db/nets64_vega1/NVDA_network1.txt",
                    "C:/_db/nets64_vega1/GS_network1.txt,C:/_db/nets64_vega1/MSFT_network1.txt,C:/_db/nets64_vega1/NKE_network1.txt,C:/_db/nets64_vega1/SHW_network2.txt"},
            {"TRV", "1", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/GDXJ_network1.txt,C:/_db/nets64_vega1/HD_network1.txt,C:/_db/nets64_vega1/IBM_network2.txt,C:/_db/nets64_vega1/JNJ_network1.txt" +
                            ",C:/_db/nets64_vega1/MSFT_network2.txt,C:/_db/nets64_vega1/PG_network2.txt,C:/_db/nets64_vega1/WMT_network1.txt",
                    "C:/_db/nets64_vega1/AXP_network2.txt,C:/_db/nets64_vega1/COIN_network2.txt,C:/_db/nets64_vega1/CSCO_network2.txt,C:/_db/nets64_vega1/IBM_network1.txt" +
                            ",C:/_db/nets64_vega1/SHW_network2.txt"},
            {"UNH", "1", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/CRM_network2.txt,C:/_db/nets64_vega1/CSCO_network2.txt,C:/_db/nets64_vega1/GS_network1.txt,C:/_db/nets64_vega1/HON_network1.txt" +
                            ",C:/_db/nets64_vega1/IBM_network2.txt,C:/_db/nets64_vega1/MCD_network2.txt,C:/_db/nets64_vega1/NKE_network1.txt,C:/_db/nets64_vega1/NNBR_network1.txt" +
                            ",C:/_db/nets64_vega1/NVDA_network2.txt,C:/_db/nets64_vega1/SHW_network2.txt,C:/_db/nets64_vega1/WMT_network1.txt",
                    "C:/_db/nets64_vega1/GDXJ_network2.txt,C:/_db/nets64_vega1/HD_network1.txt,C:/_db/nets64_vega1/JNJ_network2.txt,C:/_db/nets64_vega1/KO_network2.txt" +
                            ",C:/_db/nets64_vega1/MMM_network1.txt,C:/_db/nets64_vega1/MPW_network1.txt,C:/_db/nets64_vega1/NKE_network2.txt,C:/_db/nets64_vega1/V_network2.txt"},
            {"V", "1", "2", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/CRM_network1.txt,C:/_db/nets64_vega1/GS_network1.txt,C:/_db/nets64_vega1/JPM_network1.txt",
                    "C:/_db/nets64_vega1/DIS_network1.txt,C:/_db/nets64_vega1/GDXJ_network1.txt,C:/_db/nets64_vega1/HON_network2.txt,C:/_db/nets64_vega1/MMM_network2.txt" +
                            ",C:/_db/nets64_vega1/NVDA_network2.txt,C:/_db/nets64_vega1/TRV_network1.txt"},
            {"VZ", "1", "1", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/COIN_network1.txt,C:/_db/nets64_vega1/JPM_network1.txt,C:/_db/nets64_vega1/MSFT_network1.txt,C:/_db/nets64_vega1/MSFT_network2.txt" +
                            ",C:/_db/nets64_vega1/SHW_network2.txt,C:/_db/nets64_vega1/WMT_network2.txt",
                    "C:/_db/nets64_vega1/CAT_network2.txt,C:/_db/nets64_vega1/MCD_network2.txt,C:/_db/nets64_vega1/PLUG_network1.txt,C:/_db/nets64_vega1/SIL_network1.txt" +
                            ",C:/_db/nets64_vega1/VZ_network2.txt"},
            {"WMT", "1", "2", TRAINING_VECTOR, "3", "64",
                    "C:/_db/nets64_vega1/AMGN_network2.txt,C:/_db/nets64_vega1/DIS_network2.txt,C:/_db/nets64_vega1/HD_network2.txt,C:/_db/nets64_vega1/SIL_network1.txt" +
                            ",C:/_db/nets64_vega1/SIL_network2.txt",
                    "C:/_db/nets64_vega1/GDXJ_network1.txt,C:/_db/nets64_vega1/HON_network2.txt,C:/_db/nets64_vega1/MSFT_network2.txt,C:/_db/nets64_vega1/V_network2.txt"}
    };

    final static String[][] MODELS_1024orig = {
            {"AAPL", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/CAT_network2.txt,C:/_db/nets/CRM_network2.txt",
                    "C:/_db/nets/HD_network1.txt,C:/_db/nets/HD_network2.txt"},
            {"AMGN", "1", "2", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/CRM_network1.txt,C:/_db/nets/MCD_network2.txt",
                    "C:/_db/nets/DIS_network1.txt,C:/_db/nets/KO_network1.txt,C:/_db/nets/MCD_network1.txt,C:/_db/nets/PG_network2.txt,C:/_db/nets/VZ_network2.txt"},
            {"AMZN", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/CAT_network1.txt,C:/_db/nets/CRM_network2.txt,C:/_db/nets/NKE_network2.txt",
                    "C:/_db/nets/GS_network2.txt,C:/_db/nets/MCD_network2.txt,C:/_db/nets/VZ_network1.txt"},
            {"AXP", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/DIS_network2.txt",
                    "C:/_db/nets/MRK_network2.txt"},
            {"BA", "2", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/CRM_network2.txt,C:/_db/nets/DIS_network2.txt,C:/_db/nets/GS_network1.txt,C:/_db/nets/JNJ_network1.txt",
                    "C:/_db/nets/CAT_network1.txt,C:/_db/nets/GS_network2.txt,C:/_db/nets/NKE_network2.txt,C:/_db/nets/VZ_network2.txt"},
            {"CAT", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/DIS_network2.txt,C:/_db/nets/PG_network1.txt",
                    "C:/_db/nets/HD_network1.txt,C:/_db/nets/HD_network2.txt,C:/_db/nets/NKE_network1.txt"},
            {"CRM", "1", "2", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/HD_network2.txt",
                    "C:/_db/nets/CAT_network2.txt,C:/_db/nets/CRM_network2.txt,C:/_db/nets/NKE_network1.txt"},
            {"CSCO", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/CRM_network2.txt",
                    "C:/_db/nets/MRK_network2.txt"},
            {"CVX", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/CAT_network1.txt,C:/_db/nets/CRM_network2.txt,C:/_db/nets/NKE_network2.txt",
                    "C:/_db/nets/VZ_network2.txt"},
            {"DIS", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/DIS_network2.txt,C:/_db/nets/NKE_network2.txt",
                    "C:/_db/nets/PG_network2.txt"},
            {"GS", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/CAT_network1.txt,C:/_db/nets/NKE_network2.txt,C:/_db/nets/VZ_network2.txt",
                    "C:/_db/nets/GS_network2.txt,C:/_db/nets/HD_network1.txt,C:/_db/nets/KO_network2.txt,C:/_db/nets/NKE_network1.txt"},
            {"HD", "1", "2", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/DIS_network1.txt,C:/_db/nets/JNJ_network2.txt,C:/_db/nets/KO_network2.txt",
                    "C:/_db/nets/DIS_network2.txt,C:/_db/nets/MRK_network2.txt,C:/_db/nets/PG_network2.txt,C:/_db/nets/VZ_network2.txt"},
            {"HON", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/DIS_network2.txt,C:/_db/nets/MCD_network1.txt",
                    "C:/_db/nets/CRM_network2.txt,C:/_db/nets/JNJ_network1.txt,C:/_db/nets/KO_network1.txt,C:/_db/nets/KO_network2.txt"},
            {"IBM", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/HD_network1.txt,C:/_db/nets/MRK_network1.txt",
                    "C:/_db/nets/MSFT_network2.txt"},
            {"JNJ", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/BA_network1.txt,C:/_db/nets/CAT_network2.txt",
                    "C:/_db/nets/CAT_network1.txt,C:/_db/nets/DIS_network2.txt,C:/_db/nets/GS_network1.txt"},
            {"JPM", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/BA_network1.txt,C:/_db/nets/JNJ_network2.txt",
                    "C:/_db/nets/KO_network1.txt,C:/_db/nets/MCD_network2.txt"},
            {"KO", "1", "3", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/BA_network1.txt,C:/_db/nets/CRM_network2.txt,C:/_db/nets/MCD_network1.txt",
                    "C:/_db/nets/DIS_network2.txt,C:/_db/nets/GS_network2.txt,C:/_db/nets/KO_network2.txt,C:/_db/nets/MRK_network2.txt"},
            {"MCD", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/CRM_network2.txt,C:/_db/nets/MCD_network1.txt,C:/_db/nets/NKE_network2.txt,C:/_db/nets/PG_network2.txt",
                    "C:/_db/nets/GS_network2.txt,C:/_db/nets/VZ_network2.txt"},
            {"MMM", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/CAT_network2.txt,C:/_db/nets/CRM_network2.txt,C:/_db/nets/HD_network1.txt,C:/_db/nets/MRK_network2.txt",
                    "C:/_db/nets/GS_network2.txt,C:/_db/nets/NKE_network1.txt,C:/_db/nets/PG_network1.txt"},
            {"MRK", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/BA_network1.txt,C:/_db/nets/CRM_network1.txt,C:/_db/nets/MRK_network2.txt",
                    "C:/_db/nets/KO_network2.txt"},
            {"MSFT", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/CRM_network2.txt,C:/_db/nets/GS_network2.txt,C:/_db/nets/NKE_network2.txt",
                    "C:/_db/nets/DIS_network2.txt,C:/_db/nets/HD_network1.txt,C:/_db/nets/MRK_network2.txt"},
            {"NKE", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/CRM_network2.txt",
                    "C:/_db/nets/KO_network2.txt"},
            {"NVDA", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/BA_network2.txt,C:/_db/nets/HD_network1.txt",
                    "C:/_db/nets/PG_network1.txt"},
            {"PG", "1", "2", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/JNJ_network1.txt",
                    "C:/_db/nets/BA_network1.txt,C:/_db/nets/CAT_network2.txt,C:/_db/nets/CRM_network1.txt,C:/_db/nets/MRK_network2.txt"},
            {"SHW", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/CRM_network2.txt,C:/_db/nets/DIS_network1.txt",
                    "C:/_db/nets/CAT_network2.txt,C:/_db/nets/VZ_network1.txt"},
            {"TRV", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/BA_network1.txt,C:/_db/nets/DIS_network2.txt,C:/_db/nets/MRK_network2.txt",
                    "C:/_db/nets/NKE_network2.txt"},
            {"UNH", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/GS_network1.txt,C:/_db/nets/JNJ_network1.txt,C:/_db/nets/PG_network2.txt",
                    "C:/_db/nets/DIS_network1.txt,C:/_db/nets/VZ_network2.txt"},
            {"V", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/CRM_network2.txt,C:/_db/nets/GS_network1.txt,C:/_db/nets/GS_network2.txt",
                    "C:/_db/nets/NKE_network2.txt,C:/_db/nets/VZ_network1.txt"},
            {"VZ", "1", "1", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/DIS_network1.txt,C:/_db/nets/DIS_network2.txt",
                    "C:/_db/nets/GS_network2.txt,C:/_db/nets/NKE_network2.txt"},
            {"WMT", "1", "3", TRAINING_VECTOR, "3", "1024",
                    "C:/_db/nets/CRM_network2.txt,C:/_db/nets/GS_network2.txt,C:/_db/nets/PG_network1.txt",
                    "C:/_db/nets/BA_network1.txt,C:/_db/nets/DIS_network2.txt,C:/_db/nets/KO_network2.txt,C:/_db/nets/NKE_network2.txt"}
    };

    String[][] MODELS_603twc3 = {
            {"GS", "1", "1", TRAINING_VECTOR, "3", "603",
                    "C:/_db/nets603_twc3/AXP_network1.txt,C:/_db/nets603_twc3/GS_network1.txt,C:/_db/nets603_twc3/HON_network1.txt",
                    "C:/_db/nets603_twc3/AAPL_network1.txt,C:/_db/nets603_twc3/CSCO_network1.txt,C:/_db/nets603_twc3/GDXJ_network1.txt"},
            {"MSFT", "1", "1", TRAINING_VECTOR, "3", "603",
                    "C:/_db/nets603_twc3/AXP_network2.txt,C:/_db/nets603_twc3/JPM_network1.txt,C:/_db/nets603_twc3/MCD_network2.txt,C:/_db/nets603_twc3/SIL_network1.txt",
                    "C:/_db/nets603_twc3/GDXJ_network1.txt,C:/_db/nets603_twc3/NVDA_network2.txt,C:/_db/nets603_twc3/V_network1.txt"},
            {"WMT", "1", "1", TRAINING_VECTOR, "3", "603",
                    "C:/_db/nets603_twc3/AMGN_network2.txt,C:/_db/nets603_twc3/CAT_network2.txt,C:/_db/nets603_twc3/COIN_network1.txt,C:/_db/nets603_twc3/GDXJ_network2.txt,C:/_db/nets603_twc3/V_network2.txt",
                    "C:/_db/nets603_twc3/MSFT_network2.txt"},
    };
}