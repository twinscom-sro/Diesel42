package predictorTasks;

import datamodels.StockDataSet;
import optimizationTasks.ModelMixer;
import org.bson.Document;
import trainingTasks.TrainingProcessor;

public class BacktestingProcessor {

    public StockDataSet stockData;
    String tickerId;
    String scenario;
    public int totalDays;

    String[] message;
    String[] date;
    String[] signal;
    double[] price;
    int[] qty;
    double[] profit;
    double[] gain;
    int[] transactions;
    double[] value;

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

        message = new String[totalDays];
        date = new String[totalDays];
        signal = new String[totalDays];
        price = new double[totalDays];
        qty = new int[totalDays];
        profit = new double[totalDays];
        gain = new double[totalDays];
        transactions = new int[totalDays];
        value = new double[totalDays];

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
            message[d] = "";
            signal[d]=" ";
            date[d] = stockData.dates[d];
            price[d] = stockData.price[d];
            boolean buySignal = buySignals[d]=='1' || buySignals[d]=='B';
            boolean sellSignal = sellSignals[d]=='1' || sellSignals[d]=='S';
            if( buySignal && sellSignal ){
                buySignal = false;
                sellSignal = false;
                message[d] = "Mixed signals detected";
                signal[d]="X";
            }
            if( buySignal ) signal[d]="B";
            if( sellSignal ) signal[d]="S";
            if( position==0 &&  buySignal ){
                position=1;
                purchasePrice = close[d+1];
                message[d] = String.format("Buy @ %.2f", purchasePrice);

            }else if( position==1 && sellSignal ){
                position=0;
                trades++;
                totalProfit += close[d+1]-purchasePrice;
                totalGain += ((close[d+1]-purchasePrice)/purchasePrice)*100;
                message[d] = String.format("Sell @ %.2f, profit=%.2f, gain=%.2f", close[d+1],
                        (close[d+1]-purchasePrice), (((close[d+1]-purchasePrice)/purchasePrice)*100));
                signal[d]="S";
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
            qty[d] = position;
            profit[d] = totalProfit;
            gain[d] = totalGain;
            transactions[d] = trades;
            value[d] = position*stockData.price[d]+totalProfit;
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
                        String.format("[%s, %.2f, %.2f, %d, %.2f, %.2f, %d, %.2f, %.2f, %d, %.2f, %.2f, %.2f]",
                                tickerId, winRate, rrRatio, winTrades, totalWinAmount, avgWin,
                                lossTrades, totalLossAmount, avgLoss, trades, totalProfit, totalGain, avgGain) );
        //String buf = results.toJson();
       // System.out.println("Results:");
       // System.out.println(buf);

        return results;
    }//backtesting

    public void writeTradingHistoryDetail(StringBuilder results, ModelMixer mx) {
        results.append("\nDATES,PRICE,SIGNAL,QTY,VALUE,TRADES,PROFIT,GAIN,MESSAGE\n");
        for (int d = 0; d < totalDays; d++) {
            String mxSignal="";
            if( mx!=null ){
                mxSignal = ","+mx.signalPattern[d];
            }
            if( date[d]!=null ) results.append(String.format("%10s,%6.2f%s,%2s,%3d,%8.2f,%3d,%8.2f,%8.2f,%s\n",
                    date[d], price[d], mxSignal, signal[d], qty[d], value[d], transactions[d], profit[d], gain[d], message[d] ) );
        }
    }


}
