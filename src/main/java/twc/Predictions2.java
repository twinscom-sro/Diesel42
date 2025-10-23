package twc;

import environment.Utilities;
import forecast2.Forecast;
import optimizationTasks.ModelMixer;
import org.bson.Document;
import org.nd4j.linalg.dataset.DataSet;
import predictorTasks.BacktestingProcessor;

import java.util.ArrayList;
import java.util.List;

public class Predictions2 {

    final static String TRAINING_VECTOR = "cmf,obv,willR,atrPct,kcMPct,kcUPct,macdv,macdvSignal";

    public static void main(String[] args) {
        String KPI = "C:/_db/kpis/";
        String OUT = "C:/_arcturus/2025-10-22b/";
        String[][][] MODEL_DB = {MODELS_120};
        String outFile = "C:/_arcturus/2025-10-22b/summary.txt";

        StringBuilder out = new StringBuilder();
        for (int i = 0; i < MODEL_DB.length; i++) {
            String[][] MODELS = MODEL_DB[i];

            for (String[] model : MODELS) {
                String tickerId = model[0];
                int buyThreshold = Integer.parseInt(model[1]);
                int sellThreshold = Integer.parseInt(model[2]);
                String[] config = model[3].split(",");
                int multiplier = 5;//Integer.parseInt(model[4]);
                int neurons = Integer.parseInt(model[5]);
                String[] params ={"closeMA200","closeMA200xo","closeMA50","closeMA50xo",
                        "cmf","macd","macdSignal","atrDaily","atr","atrPct",
                        "mfi","pvo","obv","willR",
                        "kcLwr","kcMid","KcUpr","kcLPct","kcMPct","kcUPct",
                        "macdv","macdvSignal","mPhase","mDir"};
                String[] buyModels = model[6].split(",");
                String[] sellModels = model[7].split(",");
                //String[] periods = {"2019","2020","2021","2022", "2023"};
                String[] periods = {"2024", "2025"};

                backtestModels2(out, tickerId, KPI, OUT,
                        buyModels, sellModels, buyThreshold, sellThreshold,
                        periods, params, multiplier);
                //break;
            }
        }
        Utilities.writeFile(outFile, out);
    }

    private static Document backtestModels2(
            StringBuilder out, String tickerId, String KPIS, String OUT,
            String[] buyModels, String[] sellModels, int buyThreshold, int sellThreshold,
            String[] periods, String[] params, int multiplier)
    {
            String kpiFile = KPIS + tickerId + "_kpis.txt";
            String outFile = OUT + String.format("%s_%s_forecast.txt", tickerId, Utilities.getTimeTag() );
            //String signalsFile = OUT + String.format("%s_%s_matrix.txt", tickerId, Utilities.getTimeTag() );
            StringBuilder sb1 = new StringBuilder();

            BacktestingProcessor bp = new BacktestingProcessor(tickerId, String.format("backtest for %s", tickerId));
            bp.loadDataSet(kpiFile, periods, params, multiplier);
            List<String> modelFiles = new ArrayList<>();
            char[] modelVector = new char[buyModels.length+sellModels.length];
            int i = 0;
            while( i<buyModels.length ){
                modelFiles.add(buyModels[i]);
                modelVector[i++]='B';
            }
            for( int j=0; j<sellModels.length; j++ ){
                modelFiles.add(sellModels[j]);
                modelVector[i++]='S';
            }
            ModelMixer mx = new ModelMixer(modelVector.length, bp.totalDays);
            Forecast f = new Forecast(bp.totalDays, 120);
            DataSet ds = f.loadDataSet( bp );

            for (int k = 0; k < modelVector.length; k++) {
                //System.out.format("Model: %d, type: %s, name: %s\n",k,modelVector[k], modelFiles.get(k) );
                if( modelVector[k] == 'B' ){
                    mx.loadPredictions3(ds, k, modelFiles.get(k), 1); // buys
                }else {
                    mx.loadPredictions3(ds, k, modelFiles.get(k), 2); // sells
                }
            }
            mx.recalculateSignals(modelVector, buyThreshold, sellThreshold);
            /*for( int d=0; d<bp.totalDays; d++ ){
                System.out.format("%4d>",d);
                for( int j=0; j<modelVector.length; j++ ){
                    System.out.print(mx.signals[j+2][d]);
                }
                System.out.println();
            }*/
            //mx.writeSignalsMatrix( sb1 );
            Document result = bp.backtesting(mx.signals[0], mx.signals[1]);
            sb1.append( result.toJson() ).append("\n\n--------------------------------------------\n");
            System.out.println(result.toJson());
            bp.writeTradingHistoryDetail(sb1,mx);
            sb1.append("\n--------------------------------------------\n");
            mx.writeForecastDetailMatrix(bp,sb1);
            Utilities.writeFile(outFile, sb1);
            out.append( result.toJson() ).append(",\n");
            return result;
    }


    final static String[][] MODELS_120 = {
            {"PLUG", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/JPM_15A__buys.txt,c:/_db/nets_120/AMGN_14A__buys.txt",
                    "c:/_db/nets_120/HON_15A__buys.txt"},
            {"NVDA", "1", "3", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/GS_14A__buys.txt,c:/_db/nets_120/NVDA_14A__buys.txt,c:/_db/nets_120/AXP_15A__buys.txt,c:/_db/nets_120/V_15A__buys.txt",
                    "c:/_db/nets_120/AMGN_14A__buys.txt,c:/_db/nets_120/CRM_14A__buys.txt,c:/_db/nets_120/PG_15A__buys.txt,c:/_db/nets_120/V_14A__buys.txt"},
            {"GDXJ", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/GDXJ_14A__buys.txt,c:/_db/nets_120/PLUG_14A__buys.txt,c:/_db/nets_120/AMZN_15A__buys.txt,c:/_db/nets_120/CVX_15A__buys.txt,c:/_db/nets_120/HD_15A__buys.txt,c:/_db/nets_120/MMM_14A__buys.txt,c:/_db/nets_120/PG_15A__buys.txt,c:/_db/nets_120/SHW_14A__buys.txt,c:/_db/nets_120/V_14A__buys.txt",
                    "c:/_db/nets_120/AMZN_15A__buys.txt,c:/_db/nets_120/UNH_15A__buys.txt,c:/_db/nets_120/AXP_14A__buys.txt,c:/_db/nets_120/GDXJ_14A__buys.txt,c:/_db/nets_120/NKE_14A__buys.txt,c:/_db/nets_120/NKE_15A__buys.txt"},
            {"MSTR", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/MSTR_14A__buys.txt,c:/_db/nets_120/JPM_14A__buys.txt,c:/_db/nets_120/MCD_14A__buys.txt,c:/_db/nets_120/MMM_14A__buys.txt",
                    "c:/_db/nets_120/MRK_14A__buys.txt,c:/_db/nets_120/AXP_15A__buys.txt,c:/_db/nets_120/BA_14A__buys.txt,c:/_db/nets_120/GS_14A__buys.txt,c:/_db/nets_120/GS_15A__buys.txt,c:/_db/nets_120/IBM_14A__buys.txt,c:/_db/nets_120/MCD_15A__buys.txt,c:/_db/nets_120/MMM_15A__buys.txt"},
            {"MPW", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/MPW_14A__buys.txt,c:/_db/nets_120/AMZN_14A__buys.txt,c:/_db/nets_120/MCD_14A__buys.txt,c:/_db/nets_120/MMM_14A__buys.txt,c:/_db/nets_120/NKE_14A__buys.txt,c:/_db/nets_120/V_14A__buys.txt",
                    "c:/_db/nets_120/AXP_15A__buys.txt,c:/_db/nets_120/CVX_14A__buys.txt,c:/_db/nets_120/MPW_14A__buys.txt,c:/_db/nets_120/MRK_14A__buys.txt"},
            {"NNBR", "3", "2", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/GS_14A__buys.txt,c:/_db/nets_120/NNBR_14A__buys.txt,c:/_db/nets_120/TRV_15A__buys.txt,c:/_db/nets_120/CAT_14A__buys.txt,c:/_db/nets_120/CSCO_14A__buys.txt,c:/_db/nets_120/JNJ_15A__buys.txt,c:/_db/nets_120/JPM_14A__buys.txt,c:/_db/nets_120/MMM_14A__buys.txt,c:/_db/nets_120/SHW_14A__buys.txt",
                    "c:/_db/nets_120/CSCO_15A__buys.txt,c:/_db/nets_120/UNH_15A__buys.txt,c:/_db/nets_120/AMZN_15A__buys.txt,c:/_db/nets_120/CSCO_15A__buys.txt,c:/_db/nets_120/GS_15A__buys.txt,c:/_db/nets_120/NNBR_14A__buys.txt,c:/_db/nets_120/PG_15A__buys.txt,c:/_db/nets_120/UNH_15A__buys.txt"},
            {"COIN", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/COIN_14A__buys.txt,c:/_db/nets_120/CSCO_14A__buys.txt,c:/_db/nets_120/GS_15A__buys.txt",
                    "c:/_db/nets_120/COIN_14A__buys.txt,c:/_db/nets_120/CVX_14A__buys.txt,c:/_db/nets_120/IBM_14A__buys.txt"},
            {"VZ", "2", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/GS_14A__buys.txt,c:/_db/nets_120/MSFT_15A__buys.txt,c:/_db/nets_120/VZ_15A__buys.txt",
                    "c:/_db/nets_120/AMZN_14A__buys.txt,c:/_db/nets_120/NNBR_14A__buys.txt,c:/_db/nets_120/AMGN_14A__buys.txt,c:/_db/nets_120/VZ_14A__buys.txt"},
            {"KO", "1", "2", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/GS_15A__buys.txt,c:/_db/nets_120/KO_14A__buys.txt,c:/_db/nets_120/MMM_15A__buys.txt,c:/_db/nets_120/AMZN_14A__buys.txt,c:/_db/nets_120/AMZN_15A__buys.txt,c:/_db/nets_120/MRK_14A__buys.txt,c:/_db/nets_120/WMT_15A__buys.txt",
                    "c:/_db/nets_120/AMGN_14A__buys.txt,c:/_db/nets_120/AXP_14A__buys.txt,c:/_db/nets_120/JPM_14A__buys.txt,c:/_db/nets_120/MMM_14A__buys.txt,c:/_db/nets_120/MPW_14A__buys.txt,c:/_db/nets_120/MSTR_14A__buys.txt,c:/_db/nets_120/NVDA_14A__buys.txt,c:/_db/nets_120/GDXJ_14A__buys.txt,c:/_db/nets_120/KO_14A__buys.txt"},
            {"NKE", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/NKE_15A__buys.txt,c:/_db/nets_120/PLUG_14A__buys.txt,c:/_db/nets_120/KO_15A__buys.txt,c:/_db/nets_120/WMT_14A__buys.txt",
                    "c:/_db/nets_120/NKE_15A__buys.txt"},
            {"CSCO", "2", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/CSCO_14A__buys.txt,c:/_db/nets_120/CSCO_15A__buys.txt,c:/_db/nets_120/CVX_14A__buys.txt,c:/_db/nets_120/DIS_14A__buys.txt,c:/_db/nets_120/GS_15A__buys.txt,c:/_db/nets_120/BA_15A__buys.txt,c:/_db/nets_120/COIN_14A__buys.txt,c:/_db/nets_120/CVX_14A__buys.txt,c:/_db/nets_120/DIS_14A__buys.txt,c:/_db/nets_120/JPM_14A__buys.txt,c:/_db/nets_120/MPW_14A__buys.txt,c:/_db/nets_120/NVDA_14A__buys.txt",
                    "c:/_db/nets_120/AMGN_14A__buys.txt,c:/_db/nets_120/CSCO_14A__buys.txt,c:/_db/nets_120/MSTR_14A__buys.txt"},
            {"MRK", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/MRK_14A__buys.txt,c:/_db/nets_120/V_15A__buys.txt,c:/_db/nets_120/WMT_15A__buys.txt",
                    "c:/_db/nets_120/UNH_15A__buys.txt,c:/_db/nets_120/MRK_15A__buys.txt"},
            {"DIS", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/DIS_15A__buys.txt,c:/_db/nets_120/HD_14A__buys.txt,c:/_db/nets_120/KO_14A__buys.txt,c:/_db/nets_120/PLUG_14A__buys.txt,c:/_db/nets_120/SHW_15A__buys.txt",
                    "c:/_db/nets_120/MRK_14A__buys.txt,c:/_db/nets_120/WMT_14A__buys.txt,c:/_db/nets_120/DIS_14A__buys.txt,c:/_db/nets_120/NVDA_15A__buys.txt"},
            {"PG", "2", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/COIN_14A__buys.txt,c:/_db/nets_120/CRM_14A__buys.txt,c:/_db/nets_120/DIS_14A__buys.txt,c:/_db/nets_120/NVDA_14A__buys.txt,c:/_db/nets_120/WMT_15A__buys.txt,c:/_db/nets_120/BA_15A__buys.txt,c:/_db/nets_120/CAT_15A__buys.txt,c:/_db/nets_120/CRM_14A__buys.txt,c:/_db/nets_120/CVX_14A__buys.txt,c:/_db/nets_120/HON_15A__buys.txt,c:/_db/nets_120/MRK_14A__buys.txt,c:/_db/nets_120/UNH_15A__buys.txt,c:/_db/nets_120/V_15A__buys.txt,c:/_db/nets_120/WMT_15A__buys.txt",
                    "c:/_db/nets_120/CSCO_15A__buys.txt,c:/_db/nets_120/MSFT_14A__buys.txt,c:/_db/nets_120/PG_14A__buys.txt,c:/_db/nets_120/V_14A__buys.txt"},
            {"CVX", "2", "2", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/CVX_14A__buys.txt,c:/_db/nets_120/CVX_15A__buys.txt,c:/_db/nets_120/GDXJ_14A__buys.txt,c:/_db/nets_120/GS_15A__buys.txt,c:/_db/nets_120/HON_15A__buys.txt",
                    "c:/_db/nets_120/CSCO_14A__buys.txt,c:/_db/nets_120/HD_15A__buys.txt,c:/_db/nets_120/CVX_15A__buys.txt,c:/_db/nets_120/VZ_14A__buys.txt"},
            {"MMM", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/MMM_14A__buys.txt,c:/_db/nets_120/MSTR_14A__buys.txt,c:/_db/nets_120/CSCO_15A__buys.txt,c:/_db/nets_120/MSFT_14A__buys.txt",
                    "c:/_db/nets_120/PG_15A__buys.txt,c:/_db/nets_120/WMT_15A__buys.txt,c:/_db/nets_120/MMM_15A__buys.txt"},
            {"JNJ", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/HON_15A__buys.txt,c:/_db/nets_120/JNJ_14A__buys.txt,c:/_db/nets_120/BA_14A__buys.txt,c:/_db/nets_120/MMM_15A__buys.txt,c:/_db/nets_120/SHW_14A__buys.txt",
                    "c:/_db/nets_120/AXP_15A__buys.txt,c:/_db/nets_120/NVDA_14A__buys.txt,c:/_db/nets_120/WMT_15A__buys.txt,c:/_db/nets_120/JNJ_15A__buys.txt,c:/_db/nets_120/MSFT_15A__buys.txt"},
            {"HON", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/HON_15A__buys.txt,c:/_db/nets_120/IBM_15A__buys.txt,c:/_db/nets_120/NKE_14A__buys.txt",
                    "c:/_db/nets_120/IBM_14A__buys.txt,c:/_db/nets_120/KO_14A__buys.txt,c:/_db/nets_120/HON_14A__buys.txt,c:/_db/nets_120/PLUG_14A__buys.txt"},
            {"AMZN", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/AMZN_15A__buys.txt,c:/_db/nets_120/KO_14A__buys.txt",
                    "c:/_db/nets_120/AMZN_14A__buys.txt"},
            {"BA", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/BA_15A__buys.txt,c:/_db/nets_120/CSCO_15A__buys.txt,c:/_db/nets_120/IBM_15A__buys.txt",
                    "c:/_db/nets_120/BA_15A__buys.txt"},
            {"CRM", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/CRM_14A__buys.txt,c:/_db/nets_120/CRM_15A__buys.txt",
                    "c:/_db/nets_120/VZ_14A__buys.txt,c:/_db/nets_120/WMT_14A__buys.txt,c:/_db/nets_120/CRM_15A__buys.txt"},
            {"AAPL", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/AAPL_14A__buys.txt,c:/_db/nets_120/IBM_14A__buys.txt,c:/_db/nets_120/HD_14A__buys.txt",
                    "c:/_db/nets_120/UNH_15A__buys.txt,c:/_db/nets_120/AAPL_15A__buys.txt"},
            {"TRV", "2", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/AXP_15A__buys.txt,c:/_db/nets_120/CAT_14A__buys.txt,c:/_db/nets_120/MCD_14A__buys.txt,c:/_db/nets_120/TRV_14A__buys.txt,c:/_db/nets_120/AAPL_15A__buys.txt,c:/_db/nets_120/NKE_14A__buys.txt,c:/_db/nets_120/SHW_15A__buys.txt,c:/_db/nets_120/WMT_15A__buys.txt",
                    "c:/_db/nets_120/NVDA_14A__buys.txt,c:/_db/nets_120/TRV_15A__buys.txt"},
            {"IBM", "1", "2", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/IBM_14A__buys.txt,c:/_db/nets_120/IBM_15A__buys.txt,c:/_db/nets_120/MSTR_14A__buys.txt,c:/_db/nets_120/CSCO_15A__buys.txt,c:/_db/nets_120/HD_15A__buys.txt,c:/_db/nets_120/NKE_14A__buys.txt,c:/_db/nets_120/NKE_15A__buys.txt",
                    "c:/_db/nets_120/DIS_14A__buys.txt,c:/_db/nets_120/MRK_14A__buys.txt,c:/_db/nets_120/IBM_14A__buys.txt,c:/_db/nets_120/IBM_15A__buys.txt,c:/_db/nets_120/NVDA_14A__buys.txt,c:/_db/nets_120/VZ_14A__buys.txt,c:/_db/nets_120/WMT_15A__buys.txt"},
            {"AMGN", "2", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/AMGN_14A__buys.txt,c:/_db/nets_120/AMGN_15A__buys.txt,c:/_db/nets_120/MCD_14A__buys.txt,c:/_db/nets_120/VZ_14A__buys.txt,c:/_db/nets_120/HON_14A__buys.txt,c:/_db/nets_120/V_15A__buys.txt",
                    "c:/_db/nets_120/CVX_15A__buys.txt,c:/_db/nets_120/SHW_14A__buys.txt,c:/_db/nets_120/AMGN_15A__buys.txt,c:/_db/nets_120/MCD_15A__buys.txt"},
            {"MCD", "1", "2", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/AAPL_14A__buys.txt,c:/_db/nets_120/MCD_14A__buys.txt,c:/_db/nets_120/NVDA_15A__buys.txt,c:/_db/nets_120/KO_14A__buys.txt,c:/_db/nets_120/MRK_15A__buys.txt,c:/_db/nets_120/WMT_14A__buys.txt",
                    "c:/_db/nets_120/AMZN_14A__buys.txt,c:/_db/nets_120/AXP_14A__buys.txt,c:/_db/nets_120/COIN_15A__buys.txt,c:/_db/nets_120/CSCO_15A__buys.txt,c:/_db/nets_120/PG_15A__buys.txt,c:/_db/nets_120/VZ_14A__buys.txt,c:/_db/nets_120/VZ_15A__buys.txt,c:/_db/nets_120/MCD_14A__buys.txt,c:/_db/nets_120/VZ_14A__buys.txt"},
            {"JPM", "1", "2", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/JPM_14A__buys.txt,c:/_db/nets_120/MSTR_14A__buys.txt",
                    "c:/_db/nets_120/AAPL_14A__buys.txt,c:/_db/nets_120/AMZN_14A__buys.txt,c:/_db/nets_120/CSCO_14A__buys.txt,c:/_db/nets_120/AXP_15A__buys.txt,c:/_db/nets_120/HD_14A__buys.txt,c:/_db/nets_120/JPM_14A__buys.txt,c:/_db/nets_120/JPM_15A__buys.txt"},
            {"AXP", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/AXP_15A__buys.txt,c:/_db/nets_120/IBM_15A__buys.txt",
                    "c:/_db/nets_120/AXP_14A__buys.txt,c:/_db/nets_120/CSCO_15A__buys.txt,c:/_db/nets_120/HON_14A__buys.txt"},
            {"SHW", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/SHW_15A__buys.txt,c:/_db/nets_120/AAPL_15A__buys.txt,c:/_db/nets_120/CAT_15A__buys.txt,c:/_db/nets_120/GDXJ_14A__buys.txt,c:/_db/nets_120/MRK_14A__buys.txt,c:/_db/nets_120/WMT_14A__buys.txt",
                    "c:/_db/nets_120/SHW_15A__buys.txt"},
            {"V", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/V_15A__buys.txt,c:/_db/nets_120/CRM_14A__buys.txt,c:/_db/nets_120/IBM_14A__buys.txt,c:/_db/nets_120/IBM_15A__buys.txt,c:/_db/nets_120/JPM_14A__buys.txt,c:/_db/nets_120/MCD_15A__buys.txt,c:/_db/nets_120/MRK_15A__buys.txt",
                    "c:/_db/nets_120/DIS_14A__buys.txt,c:/_db/nets_120/PLUG_14A__buys.txt,c:/_db/nets_120/UNH_14A__buys.txt,c:/_db/nets_120/V_15A__buys.txt"},
            {"UNH", "2", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/AMZN_15A__buys.txt,c:/_db/nets_120/GS_15A__buys.txt,c:/_db/nets_120/MSTR_14A__buys.txt,c:/_db/nets_120/UNH_14A__buys.txt,c:/_db/nets_120/UNH_15A__buys.txt,c:/_db/nets_120/BA_15A__buys.txt,c:/_db/nets_120/CRM_15A__buys.txt,c:/_db/nets_120/MCD_15A__buys.txt,c:/_db/nets_120/MMM_14A__buys.txt,c:/_db/nets_120/MRK_15A__buys.txt",
                    "c:/_db/nets_120/AAPL_14A__buys.txt,c:/_db/nets_120/WMT_14A__buys.txt,c:/_db/nets_120/MSTR_14A__buys.txt,c:/_db/nets_120/UNH_14A__buys.txt"},
            {"HD", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/CSCO_15A__buys.txt,c:/_db/nets_120/HD_14A__buys.txt,c:/_db/nets_120/PLUG_14A__buys.txt",
                    "c:/_db/nets_120/IBM_15A__buys.txt,c:/_db/nets_120/PG_14A__buys.txt,c:/_db/nets_120/HD_14A__buys.txt"},
            {"CAT", "2", "2", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/CAT_14A__buys.txt,c:/_db/nets_120/CAT_15A__buys.txt,c:/_db/nets_120/GDXJ_14A__buys.txt,c:/_db/nets_120/AXP_15A__buys.txt,c:/_db/nets_120/CRM_14A__buys.txt,c:/_db/nets_120/CSCO_15A__buys.txt,c:/_db/nets_120/DIS_14A__buys.txt,c:/_db/nets_120/HD_14A__buys.txt,c:/_db/nets_120/NKE_14A__buys.txt",
                    "c:/_db/nets_120/MCD_15A__buys.txt,c:/_db/nets_120/WMT_14A__buys.txt,c:/_db/nets_120/AMZN_15A__buys.txt,c:/_db/nets_120/CAT_14A__buys.txt,c:/_db/nets_120/CAT_15A__buys.txt"},
            {"MSFT", "2", "2", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/AMZN_15A__buys.txt,c:/_db/nets_120/CRM_14A__buys.txt,c:/_db/nets_120/MSFT_14A__buys.txt,c:/_db/nets_120/MSFT_15A__buys.txt",
                    "c:/_db/nets_120/CSCO_14A__buys.txt,c:/_db/nets_120/MSFT_14A__buys.txt,c:/_db/nets_120/MSFT_15A__buys.txt"},
            {"GS", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/GS_14A__buys.txt,c:/_db/nets_120/MRK_14A__buys.txt,c:/_db/nets_120/MSTR_14A__buys.txt,c:/_db/nets_120/MCD_15A__buys.txt,c:/_db/nets_120/MRK_15A__buys.txt",
                    "c:/_db/nets_120/VZ_14A__buys.txt,c:/_db/nets_120/CSCO_15A__buys.txt,c:/_db/nets_120/GS_15A__buys.txt"},
            {"WMT", "1", "1", TRAINING_VECTOR, "3", "4096",
                    "c:/_db/nets_120/DIS_14A__buys.txt,c:/_db/nets_120/WMT_14A__buys.txt,c:/_db/nets_120/AXP_15A__buys.txt,c:/_db/nets_120/BA_14A__buys.txt,c:/_db/nets_120/CVX_15A__buys.txt,c:/_db/nets_120/SHW_15A__buys.txt",
                    "c:/_db/nets_120/NNBR_14A__buys.txt,c:/_db/nets_120/NVDA_14A__buys.txt,c:/_db/nets_120/WMT_14A__buys.txt"}

    };

}
