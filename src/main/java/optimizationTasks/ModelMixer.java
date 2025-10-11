package optimizationTasks;

import neural.DeepLayer;
import org.bson.Document;
import predictorTasks.BacktestingProcessor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ModelMixer {
    char[][] signals;
    int totalDays;
    int numModels;
    public char[] includeVector;
    boolean optimizationFlag;

    public double maxGain;
    public int optBuy;
    public int optSell;
    public int buyThreshold;
    public int sellThreshold;
    String[] modelFile;

    public ModelMixer(int columns, int tDays) {
        numModels = columns;
        totalDays = tDays;
        signals = new char[numModels+2][totalDays];
        includeVector = new char[numModels];
        modelFile = new String[numModels];
    }

    public void loadPredictions(BacktestingProcessor bp, int pos, String netFile, int neurons ) {
        DeepLayer nn = new DeepLayer(bp.stockData.inputSize, neurons, 1);
        nn.readTopology(netFile);
        for (int d = 0; d < totalDays; d++) {
            double[] y = nn.feedForward(bp.stockData.inputVector[d]);
            signals[pos+2][d] = y[0]>0.8 ? '1' : '0';
        }
        modelFile[pos]=netFile;
    }

    public void writeFinalPredictions(BacktestingProcessor bp, StringBuilder sb) {
        for (int d = 0; d < totalDays; d++) {
            sb.append( String.format("%10s, %.2f, %.2f, %d, %d\n",
                    bp.stockData.dates[d],
                    bp.stockData.price[d], bp.stockData.zigZag[d],
                    (signals[0][d]=='B'?1:0),(signals[1][d]=='S'?1:0)) );
        }
    }


    public void startOptimization(BacktestingProcessor bp) {
        for( int i = 0; i<numModels; i++ ){ includeVector[i]='-'; }
        recalculateSignals(includeVector,1,1);
        maxGain=0;
        buyThreshold=1;
        sellThreshold=1;
        char[] optVector = null;
        int counter=0;
        for( int mBuy=0; mBuy<numModels/2; mBuy++ ){
            for( int mSell=numModels/2; mSell<numModels; mSell++ ) {
                if( mBuy==mSell ) continue;
               /* if( counter++%100 == 0){
                    System.out.format("Loop@%d(%d,%d) max=%.2f\n",counter,mBuy,mSell,maxGain);
                }*/
                char[] newVector = new char[numModels];
                System.arraycopy(includeVector, 0, newVector, 0, numModels);
                newVector[mBuy]='B';
                newVector[mSell]='S';
                recalculateSignals(newVector,buyThreshold,sellThreshold);

                Document result = bp.backtesting( signals[0], signals[1] );
                double totalGain = result.getDouble("totalGain");

                if( optVector==null ){
                    optVector = new char[numModels];
                    System.arraycopy(newVector, 0, optVector, 0, numModels);
                    maxGain = totalGain;
                }else if( totalGain>maxGain ){
                    System.arraycopy(newVector, 0, optVector, 0, numModels);
                    maxGain = totalGain;
                    //System.out.format("New maximum found at (%d,%d) gain=%.2f, %s\n",mBuy,mSell,totalGain, Arrays.toString(optVector));
                }
            }
        }
        optimizationFlag = true;
        if( optVector!=null ){
            System.arraycopy(optVector, 0, includeVector, 0, numModels);
            System.out.format("Final maximum gain=%.2f, %s\n",maxGain, Arrays.toString(optVector));
            recalculateSignals(includeVector,buyThreshold,sellThreshold);
        }
    }

    private void recalculateSignals(char[] _includeVector, int buyThreshold1, int sellThreshold1) {
        if( buyThreshold1<1 ) buyThreshold1=1;
        if( sellThreshold1<1 ) sellThreshold1=1;
        for( int d=0; d<totalDays; d++ ){
            int sumBuy=0;
            int sumSell=0;
            for( int i = 0; i<numModels; i++ ){
                if( signals[i+2][d]=='1' && _includeVector[i]=='B' ) sumBuy++;
                if( signals[i+2][d]=='1' && _includeVector[i]=='S' ) sumSell++;
            }
            signals[0][d] = ( sumBuy>=buyThreshold1 ) ? 'B' : '-';
            signals[1][d] = ( sumSell>=sellThreshold1 ) ? 'S' : '-';
        }
    }

    public boolean continueOptimization() {
        return optimizationFlag;
    }


    public int addNextBuySignal(BacktestingProcessor bp) {
        maxGain=0;
        optBuy=-1;
        for( int mBuy=0; mBuy<numModels/2; mBuy++ ){
            if( includeVector[mBuy] != '-' ) continue;
                char[] newVector = new char[numModels];
                System.arraycopy(includeVector, 0, newVector, 0, numModels);
                newVector[mBuy]='B';

                for( int i = buyThreshold-1; i<buyThreshold+1; i++ ){
                    for( int j = sellThreshold-1; j<sellThreshold+1; j++ ) {
                        recalculateSignals(newVector, i, j);
                        Document result = bp.backtesting(signals[0], signals[1]);
                        double totalGain = result.getDouble("totalGain");
                        if (totalGain > maxGain) {
                            maxGain = totalGain;
                            optBuy = mBuy;
                            buyThreshold = i>0 ? i : 1;
                            sellThreshold = j>0 ? j : 1;
                        }
                    }
                }
        }
        return optBuy;
    }

    public int addNextSellSignal(BacktestingProcessor bp) {
        maxGain=0;
        optSell=-1;
        for( int mSell=numModels/2; mSell<numModels; mSell++ ) {
            if (includeVector[mSell] != '-') continue;
            char[] newVector = new char[numModels];
            System.arraycopy(includeVector, 0, newVector, 0, numModels);
            newVector[mSell] = 'S';

            for( int i = buyThreshold-1; i<buyThreshold+1; i++ ){
                for( int j = sellThreshold-1; j<sellThreshold+1; j++ ) {
                    recalculateSignals(newVector, i, j);
                    Document result = bp.backtesting(signals[0], signals[1]);
                    double totalGain = result.getDouble("totalGain");
                    if (totalGain > maxGain) {
                        maxGain = totalGain;
                        optSell = mSell;
                        buyThreshold = i>0 ? i : 1;
                        sellThreshold = j>0 ? j : 1;
                    }
                }
            }
        }
        return optSell;
    }

    public void backtesting(BacktestingProcessor bp) {
    }

    public void writeSignalsMatrix(StringBuilder sb) {
        System.out.println("include vector size="+includeVector.length);
        sb.append("      |");
        for( int i =0; i<numModels; i++ ){
            sb.append(includeVector[i]);
        }
        sb.append("\n");

        for( int d = 0; d<totalDays; d++ ){
            sb.append(String.format("%4d",d)).append(signals[0][d]).append(signals[1][d]).append("<");
            for( int i = 0; i<numModels; i++ ){
                sb.append(signals[i+2][d]);
            }
            sb.append("\n");
        }

    }

    public void shakeIncludeVector(BacktestingProcessor bp) {
        // shake the model
        List<Integer> buys = new ArrayList<>();
        List<Integer> sells = new ArrayList<>();
        int k=0;
        for( char c  : includeVector){
            if( c=='B' ) buys.add(k);
            if( c=='S' ) sells.add(k);
            k++;
        }
        int eliminateBuy = (int) Math.floor( Math.random()*buys.size() );
        int eliminateSell = (int) Math.floor( Math.random()*sells.size() );

        if( buys.size()>1 && sells.size()<=1 ){
            includeVector[eliminateBuy]='-';
        }else if( buys.size()<=1 && sells.size()>1 ){
            includeVector[eliminateSell]='-';
        }else{
            if( Math.random()>=0.5 ) {
                includeVector[eliminateBuy] = '-';
            }else{
                includeVector[eliminateSell]='-';
            }
        }
        for( int i = buyThreshold-1; i<buyThreshold+1; i++ ){
            for( int j = sellThreshold-1; j<sellThreshold+1; j++ ) {
                recalculateSignals(includeVector, i, j);
                Document result = bp.backtesting(signals[0], signals[1]);
                double totalGain = result.getDouble("totalGain");
                if (totalGain > maxGain) {
                    maxGain = totalGain;
                    buyThreshold = i>0 ? i : 1;
                    sellThreshold = j>0 ? j : 1;
                }
            }
        }

    }

    public void writeModelConfig(BacktestingProcessor bp, StringBuilder sb) {
        for( int i = 0; i<numModels; i++ ){
            if( includeVector[i] == 'B' ){
                sb.append( String.format("Buy Component: %s\n", modelFile[i] ) );
            }
        }
        for( int i = 0; i<numModels; i++ ){
            if( includeVector[i] == 'S' ){
                sb.append( String.format("Sell Component: %s\n", modelFile[i] ) );
            }
        }
        sb.append( String.format("buyThreshold=%d\n", buyThreshold ) );
        sb.append( String.format("sellThreshold=%d\n", sellThreshold ) );
    }

    public Document forecast(BacktestingProcessor bp, char[] _includeVector, int _buyThreshold, int _sellThreshold){
        System.arraycopy(_includeVector, 0, includeVector, 0, numModels);
        recalculateSignals(_includeVector, _buyThreshold, _sellThreshold);
        Document result = bp.backtesting(signals[0], signals[1]);
        //double totalGain = result.getDouble("totalGain");
        return result;
    }




 /*   // create detailed output of the summary signals with pricing
    StringBuilder results = new StringBuilder();
    StringBuilder sb3 = new StringBuilder();
    StringBuilder sb4 = new StringBuilder();
        for( String tk : models ) {
        sb3.append( String.format(", b[%s]",tk) );
        sb4.append( String.format(", s[%s]",tk) );
    }
        results.append( String.format("DATES,PRICE,ZIGZAG,buy[Exp],sell[Exp],buy[Act],sell[Act]%s%s",sb3.toString(),sb4.toString()) );
    char[] buyVector = new char[totalDays];
    char[] sellVector = new char[totalDays];
    int maxSig1=0;
    int maxSig2=0;
        for (int d = 0; d < totalDays; d++) {
        int sig1=0;
        int sig2=0;
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder();
        for( int i=0; i<models.length; i++ ) {
            if (buySignal[i][d] > 0.8) sig1++;
            if (sellSignal[i][d] > 0.8) sig2++;
            sb1.append(String.format(", %6.3f", buySignal[i][d]));
            sb2.append(String.format(", %6.3f", sellSignal[i][d]));
        }
        buySignal[0][d]=sig1;
        sellSignal[0][d]=sig2;
        results.append( String.format("\n%10s, %.2f, %.2f, %.0f, %.0f, %3d, %3d%s%s",
                tp.dates[d], tp.price[d], tp.zigZag[d], tp.buySignal[d], tp.sellSignal[d],
                sig1,sig2,sb1.toString(),sb2.toString()) );
        if( sig1>maxSig1 ) maxSig1=sig1;
        if( sig2>maxSig2 ) maxSig2=sig2;
    }
    String resultsFile = OUT+String.format("%s_forecast.txt",tkrSignal);
        Utilities.writeFile(resultsFile, results);

    // create trading vector
    StringBuilder pattern = new StringBuilder();
    char[] last15days = new char[15];
    maxSig1 = 3*maxSig1/4;
    maxSig2 = 3*maxSig2/4;
        for (int d = 0; d < totalDays; d++) {
        buyVector[d]='0';
        sellVector[d]='0';
        if( buySignal[0][d] >= maxSig1 ) buyVector[d]='1';
        if( sellSignal[0][d] >= maxSig2 ) sellVector[d]='1';

        char flag='-';
        if( buySignal[0][d] > 0 ) flag='b';
        if( sellSignal[0][d] > 0 ) flag='s';
        if( buySignal[0][d] > 0 && sellSignal[0][d] > 0 ) flag='x';

        if( buyVector[d]=='1' && sellVector[d]!='1' ) flag='B';
        if( buyVector[d]!='1' && sellVector[d]=='1' ) flag='S';
        if( buyVector[d]=='1' && sellVector[d]=='1' ) flag='X';
        //if( buyVector[d]!='1' && sellVector[d]!='1' ) pattern.append('-');
        pattern.append( flag );
        if( d>=totalDays-15 ) last15days[d-(totalDays-15)]=flag;
        if( d%90 ==0 ) pattern.append('\n');
    }

    // System.out.format("\nSignal pattern:\n%s\n",pattern);

    BacktestingProcessor bp = new BacktestingProcessor( tkrSignal, String.format("simple backtest for %s",tkrSignal) );
        bp.loadDataSet(kpiFile, filters, params, multiplier);

        sb.append("last15=[").append(last15days).append("], ").append( bp.backtesting( buyVector, sellVector ).toJson() ).append('\n');
*/
}
