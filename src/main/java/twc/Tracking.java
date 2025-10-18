package twc;

import environment.Utilities;
import org.bson.Document;

import java.util.ArrayList;
import java.util.List;

public class Tracking {

    public static void main(String[] args) {
        String[][][] MODEL_DB = {
                Predictions.MODELS_1024orig,
                Predictions.MODELS_603twc3,
                Predictions.MODELS_64vega1,
                Predictions.MODELS_1024vega2  };

        //String[] tickers = {"COIN","WMT","MPW"};
        String[] tickers = {"WMT","GS","MSFT","CAT","HD","UNH",
                "V","SHW","AXP","JPM","MCD","AMGN","IBM","TRV",
                "AAPL","CRM","BA","AMZN","HON","JNJ","NVDA",
                "MMM","CVX","PG","DIS","MRK","CSCO","NKE","KO",
                "VZ","COIN","SIL","MPW","PLUG","NNBR","GDXJ","MSTR"};

        String KPI = "C:/_db/kpis/";
        String OUT = "C:/_arcturus/2025-10-15a/";
        String DASHBOARD = "C:/_arcturus/2025-10-15a/dashboard2.txt";

        StringBuilder mdb = new StringBuilder();
        List<String> output = new ArrayList<String>();
        for(String ticker : tickers) {
            for( String[][] modelSet : MODEL_DB) {
                for( String[] model : modelSet ) {
                    if( model[0].equals(ticker)) {
                        //System.out.println( ticker + " " + model[0] + " " + model[1] + " " + model[2] );
                        int buyThreshold = Integer.parseInt(model[1]);
                        int sellThreshold = Integer.parseInt(model[2]);
                        String[] config = model[3].split(",");
                        int multiplier = Integer.parseInt(model[4]);
                        int neurons = Integer.parseInt(model[5]);
                        String[] buyModels = model[6].split(",");
                        String[] sellModels = model[7].split(",");
                        String[] periods = {"2024", "2025"};

                        Document result=Predictions.backtestModels(mdb, "2025-10-14", ticker, KPI, OUT, buyModels, sellModels, buyThreshold, sellThreshold, periods, config, multiplier, neurons);

                        output.add( String.format("%s, %s, %s\n",ticker, result.get("message"), result.get("priceHistory")) );
                    }
                }
            }
        }
        Utilities.writeFile(DASHBOARD,mdb);
        for( String s : output) {
            System.out.print(s);
        }
    }
}
