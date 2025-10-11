package predictorTasks;

import datamodels.StockDataSet;
import org.bson.Document;
import trainingTasks.TrainingProcessor;

public class BacktestingProcessor {

    public StockDataSet stockData;
    String tickerId;
    String scenario;
    public int totalDays;

    public BacktestingProcessor(String tkr, String _scenario){
        scenario = _scenario;
        tickerId = tkr;
    }

    public void loadDataSet(String kpiFile, String[] params, String[] filters, int multiples){
        stockData = new StockDataSet();
        stockData.loadDataSet(kpiFile, params, filters, multiples);
        totalDays = stockData.samples;
       // System.out.format("\nLoaded backtesting data for scenrio: %s", scenario);
       // System.out.format("\ninputVector.length= %d", stockData.inputVector.length);
       // System.out.format("\ntotalDays= %d\n", totalDays);
    }

    public Document backtesting( char[] buySignals, char[] sellSignals ){

        int position=0;
        int trades=0;
        double totalProfit=0;
        double purchasePrice=0;
        double totalGain=0;
        int winTrades=0;
        int lossTrades=0;
        double totalWinAmount=0;
        double totalLossAmount=0;
        double totalWinGain=0;
        double totalLossGain=0;

        double[] close = stockData.price;

        for( int d = 0; d<totalDays-1; d++ ){
            boolean buySignal = buySignals[d]=='1' || buySignals[d]=='B';
            boolean sellSignal = sellSignals[d]=='1' || sellSignals[d]=='S';
            if( buySignal && sellSignal ){
                buySignal = false;
                sellSignal = false;
            }
            if( position==0 &&  buySignal ){
                position=1;
                purchasePrice = close[d+1];
            }else if( position==1 && sellSignal ){
                position=0;
                trades++;
                totalProfit += close[d+1]-purchasePrice;
                totalGain += ((close[d+1]-purchasePrice)/purchasePrice)*100;
                if( close[d+1]>=purchasePrice ){
                    winTrades++;
                    totalWinAmount += (close[d+1]-purchasePrice);
                    totalWinGain += ((close[d+1]-purchasePrice)/purchasePrice)*100;
                }else if( close[d+1]<purchasePrice ){
                    lossTrades++;
                    totalLossAmount +=(purchasePrice-close[d+1]);
                    totalLossGain +=((purchasePrice-close[d+1])/purchasePrice)*100;
                }
            }
        }

        double winRate = trades>0 ? winTrades*100.0/trades : 0;
        double avgWin = winTrades>0 ? totalWinGain/winTrades : 0;
        double avgLoss = lossTrades>0 ? totalLossGain/lossTrades : 0;
        double rrRatio = avgLoss!=0 ? avgWin / avgLoss : 0;
        double avgGain = trades>0 ? totalGain/trades : 0;

        Document results = new Document().append(scenario,scenario)
                .append("winRate",winRate).append("rrRatio", rrRatio )
                .append("winTrades", winTrades).append("winAmount",totalWinAmount).append("avgWin", avgWin )
                .append("lossTrades", lossTrades).append("lossAmount",totalLossAmount).append("avgLoss", avgLoss )
                .append("trades", trades).append("totalProfit",totalProfit).append("totalGain", totalGain).append("avgGain",avgGain)
                .append("csv",
                        String.format("[%s, %.2f, %.2f, %d, %.2f, %.2f, %d, %.2f, %.2f, %d, %.2f, %.2f]",
                                tickerId, winRate, rrRatio, winTrades, totalWinAmount, avgWin,
                                lossTrades, totalLossAmount, avgLoss, trades, totalProfit, totalGain) );
        //String buf = results.toJson();
       // System.out.println("Results:");
       // System.out.println(buf);

        return results;
    }//backtesting

}
