package optimizationTasks;

import environment.Utilities;
import neural.DeepLayer;
import org.bson.Document;
import predictorTasks.BacktestingProcessor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ModelMixer {
    public char[][] signals;
    char[] signalPattern;
    int totalDays;
    int numModels;
    public char[] includeVector;
    boolean optimizationFlag;

    public int buyThreshold;
    public int sellThreshold;
    String[] modelFile;
    String[] modelName;

    public ModelMixer(int columns, int tDays) {
        numModels = columns;
        totalDays = tDays;
        signals = new char[numModels + 2][totalDays];
        signalPattern = new char[totalDays];
        includeVector = new char[numModels];
        modelFile = new String[numModels];
        modelName = new String[numModels];
        modelName[0] = "BUY[Act]";
        modelName[1] = "SELL[Act]";
    }

    public void loadPredictions(BacktestingProcessor bp, int pos, String _name, String netFile, int neurons) {
        DeepLayer nn = new DeepLayer(bp.stockData.inputSize, neurons, 1);
        nn.readTopology(netFile);
        for (int d = 0; d < totalDays; d++) {
            double[] y = nn.feedForward(bp.stockData.inputVector[d]);
            signals[pos + 2][d] = y[0] > 0.8 ? '1' : '0';
        }
        modelFile[pos] = netFile;
        modelName[pos] = _name;
    }

    public void writeFinalPredictions(BacktestingProcessor bp, StringBuilder sb) {
        for (int d = 0; d < totalDays; d++) {
            sb.append(String.format("%10s, %.2f, %.2f, %d, %d\n",
                    bp.stockData.dates[d],
                    bp.stockData.price[d], bp.stockData.zigZag[d],
                    (signals[0][d] == 'B' ? 1 : 0), (signals[1][d] == 'S' ? 1 : 0)));
        }
    }


    public void startOptimization(BacktestingProcessor bp) {
        for (int i = 0; i < numModels; i++) {
            includeVector[i] = '-';
        }
        recalculateSignals(includeVector, 1, 1);
        double maxGain = 0;
        char[] optVector = null;
        int counter = 0;
        for (int mBuy = 0; mBuy < numModels / 2; mBuy++) {
            for (int mSell = numModels / 2; mSell < numModels; mSell++) {
                if (mBuy == mSell) continue;
               /* if( counter++%100 == 0){
                    System.out.format("Loop@%d(%d,%d) max=%.2f\n",counter,mBuy,mSell,maxGain);
                }*/
                char[] newVector = new char[numModels];
                System.arraycopy(includeVector, 0, newVector, 0, numModels);
                newVector[mBuy] = 'B';
                newVector[mSell] = 'S';
                recalculateSignals(newVector, 1, 1);

                Document result = bp.backtesting(signals[0], signals[1]);
                double totalGain = result.getDouble("totalGain");

                if (optVector == null) {
                    optVector = new char[numModels];
                    System.arraycopy(newVector, 0, optVector, 0, numModels);
                } else if (totalGain > maxGain) {
                    System.arraycopy(newVector, 0, optVector, 0, numModels);
                    maxGain = totalGain;
                    //System.out.format("New maximum found at (%d,%d) gain=%.2f, %s\n",mBuy,mSell,totalGain, Arrays.toString(optVector));
                }
            }
        }
        optimizationFlag = true;
        if (optVector != null) {
            System.arraycopy(optVector, 0, includeVector, 0, numModels);
            buyThreshold = 1;
            sellThreshold = 1;
            System.out.format("Initial vector maximum gain=%.2f, %s\n", maxGain, Arrays.toString(optVector));
            recalculateSignals(includeVector, buyThreshold, sellThreshold);
        }
    }

    public void recalculateSignals(char[] _includeVector, int buyThreshold1, int sellThreshold1) {
        if (buyThreshold1 < 1) buyThreshold1 = 1;
        if (sellThreshold1 < 1) sellThreshold1 = 1;
        for (int d = 0; d < totalDays; d++) {
            int sumBuy = 0;
            int sumSell = 0;
            for (int i = 0; i < numModels; i++) {
                if (signals[i + 2][d] == '1' && _includeVector[i] == 'B') sumBuy++;
                if (signals[i + 2][d] == '1' && _includeVector[i] == 'S') sumSell++;
            }
            signals[0][d] = (sumBuy >= buyThreshold1) ? 'B' : '-';
            signals[1][d] = (sumSell >= sellThreshold1) ? 'S' : '-';
        }
    }

    public void recalculateSignals() {
// no include vector given, just calculate for everything

        int maxBuySignals = 0;
        int maxSellSignals = 0;
        for (int d = 0; d < totalDays; d++) {
            int sumBuy = 0;
            int sumSell = 0;
            for (int i = 0; i < numModels; i++) {
                if (signals[i + 2][d] == '1' && i < numModels / 2) sumBuy++;
                if (signals[i + 2][d] == '1' && i >= numModels / 2) sumSell++;
            }
            if (sumBuy > maxBuySignals) maxBuySignals = sumBuy;
            if (sumSell > maxSellSignals) maxSellSignals = sumSell;
        }

        buyThreshold = maxBuySignals * 3 / 4;
        sellThreshold = maxSellSignals * 3 / 4;
        for (int d = 0; d < totalDays; d++) {
            int sumBuy = 0;
            int sumSell = 0;
            for (int i = 0; i < numModels; i++) {
                if (signals[i + 2][d] == '1' && i < numModels / 2) sumBuy++;
                if (signals[i + 2][d] == '1' && i >= numModels / 2) sumSell++;
            }
            signals[0][d] = (sumBuy >= buyThreshold) ? 'B' : '-';
            signals[1][d] = (sumSell >= sellThreshold) ? 'S' : '-';
        }

    }

    public String updateSignalPattern() {
        for (int d = 0; d < totalDays; d++) {
            int sumBuy = 0;
            int sumSell = 0;
            for (int i = 0; i < numModels; i++) {
                if (signals[i + 2][d] == '1' && i < numModels / 2) sumBuy++;
                if (signals[i + 2][d] == '1' && i >= numModels / 2) sumSell++;
            }

            char flag = '-';
            if (sumBuy > 0) flag = 'b';
            if (sumSell > 0) flag = 's';
            if (sumBuy > 0 && sumSell > 0) flag = 'x';

            if (signals[0][d] == 'B') flag = 'B';
            if (signals[1][d] == 'S') flag = 'S';
            if (signals[0][d] == 'B' && signals[1][d] == 'S') flag = 'X';

            signalPattern[d] = flag;
        }
        return new String(signalPattern);
    }


    public boolean continueOptimization() {
        return optimizationFlag;
    }


 /*   public double addNextBuySignal(BacktestingProcessor bp) {
        double _maxGain=0;
        nextBuyCandidate = -1;
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
                        if (totalGain > _maxGain || nextBuyCandidate==-1) {
                            _maxGain = totalGain;
                            nextBuyCandidate = mBuy;
                            buyThreshold = i>0 ? i : 1;
                            sellThreshold = j>0 ? j : 1;
                        }
                    }
                }
        }
        return _maxGain;
    }

    public double addNextSellSignal(BacktestingProcessor bp) {
        double _maxGain=0;
        nextSellCandidate=-1;
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
                    if (totalGain > maxGain ||  nextSellCandidate==-1) {
                        _maxGain = totalGain;
                        nextSellCandidate = mSell;
                        buyThreshold = i>0 ? i : 1;
                        sellThreshold = j>0 ? j : 1;
                    }
                }
            }
        }
        return _maxGain;
    }
  */

    public void writeSignalsMatrix(StringBuilder sb) {
        System.out.println("include vector size=" + includeVector.length);
        sb.append("      |");
        for (int i = 0; i < numModels; i++) {
            sb.append(includeVector[i]);
        }
        sb.append("\n");

        for (int d = 0; d < totalDays; d++) {
            sb.append(String.format("%4d", d)).append(signals[0][d]).append(signals[1][d]).append("<");
            for (int i = 0; i < numModels; i++) {
                sb.append(signals[i + 2][d]);
            }
            sb.append("\n");
        }

    }

    /*  public void shakeIncludeVector(BacktestingProcessor bp) {
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
              includeVector[buys.get(eliminateBuy)]='-';
              System.out.format("Removing buy signal at index: %d\n", buys.get(eliminateBuy));
          }else if( buys.size()<=1 && sells.size()>1 ){
              includeVector[sells.get(eliminateSell)]='-';
              System.out.format("Removing sell signal at index: %d\n", sells.get(eliminateSell));
          }else if( buys.size()>1 && sells.size()>1 ){
              if( Math.random()>=0.5 ) {
                  includeVector[buys.get(eliminateBuy)] = '-';
                  System.out.format("Removing buy signal at index: %d\n", buys.get(eliminateBuy));
              }else{
                  includeVector[sells.get(eliminateSell)]='-';
                  System.out.format("Removing sell signal at index: %d\n", sells.get(eliminateSell));
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
  */
    public void writeModelConfig(BacktestingProcessor bp, StringBuilder sb) {
        for (int i = 0; i < numModels; i++) {
            if (includeVector[i] == 'B') {
                sb.append(String.format("Buy Component: %s\n", modelFile[i]));
            }
        }
        for (int i = 0; i < numModels; i++) {
            if (includeVector[i] == 'S') {
                sb.append(String.format("Sell Component: %s\n", modelFile[i]));
            }
        }
        sb.append(String.format("buyThreshold=%d\n", buyThreshold));
        sb.append(String.format("sellThreshold=%d\n", sellThreshold));
    }

    public Document forecast(BacktestingProcessor bp, char[] _includeVector, int _buyThreshold, int _sellThreshold) {
        buyThreshold = _buyThreshold;
        sellThreshold = _sellThreshold;
        System.arraycopy(_includeVector, 0, includeVector, 0, numModels);
        recalculateSignals(_includeVector, _buyThreshold, _sellThreshold);
        updateSignalPattern();
        Document result = bp.backtesting(signals[0], signals[1]);
        //double totalGain = result.getDouble("totalGain");
        return result;
    }

    public void writeForecastDetailMatrix(BacktestingProcessor bp, StringBuilder sb) {
        StringBuilder results = new StringBuilder();
        StringBuilder sb3 = new StringBuilder();
        StringBuilder sb4 = new StringBuilder();
        for (int i = 0; i < numModels; i++) {
            if (i < numModels / 2) sb3.append(String.format(",%s", modelName[i]));
            if (i >= numModels / 2) sb4.append(String.format(",%s", modelName[i]));
        }
        results.append(String.format("DATES,PRICE,ZIGZAG,buy[Exp],sell[Exp],SIGNAL,buy[Act],sell[Act]%s%s", sb3.toString(), sb4.toString()));
        for (int d = 0; d < totalDays; d++) {
            StringBuilder sb1 = new StringBuilder();
            StringBuilder sb2 = new StringBuilder();
            for (int i = 0; i < numModels / 2; i++) {
                sb1.append(",").append(signals[i + 2][d]);
            }
            for (int i = numModels / 2; i < numModels; i++) {
                sb2.append(",").append(signals[i + 2][d]);
            }
            results.append(String.format("\n%10s,%.2f,%.2f,%.0f,%.0f,",
                    bp.stockData.dates[d], bp.stockData.price[d], bp.stockData.zigZag[d], bp.stockData.buySignal[d], -bp.stockData.sellSignal[d]));
            results.append(signalPattern[d]).append(",");
            results.append(signals[0][d] == 'B' ? 1 : 0).append(",").append(signals[1][d] == 'S' ? -1 : 0);
            results.append(sb1.toString()).append(sb2.toString());
        }
        sb.append(results);
        //Utilities.writeFile(resultsFile, results);
    }

    /*   public void runOptimizer1(BacktestingProcessor bp) {
           boolean addBuy=true;
           int epoch=0;
           double currentGain = maxGain;
           int ageOfLastChange=0;
           while( continueOptimization() && ageOfLastChange<20 ){
               epoch++;
               System.out.format("Epoch=%d, gain=%.2f, vector=[%s]\n",epoch,maxGain,Arrays.toString(includeVector) );
               if( addBuy ){
                   addNextBuySignal(bp);
                   if( maxGain>currentGain ) {
                       includeVector[nextBuyCandidate]='B';
                       currentGain = maxGain;
                       ageOfLastChange=0;
                       System.out.format("Adding buy model=%d (%d,%d), gain=%.2f\n",nextBuyCandidate, buyThreshold, sellThreshold, currentGain);
                   }else{
                       ageOfLastChange++;
                   }
               }else{
                   addNextSellSignal(bp);
                   if( maxGain>currentGain ) {
                       includeVector[nextSellCandidate]='S';
                       currentGain = maxGain;
                       ageOfLastChange=0;
                       System.out.format("Adding sell model=%d (%d,%d), gain=%.2f\n",nextSellCandidate, buyThreshold, sellThreshold, currentGain);
                   }else{
                       ageOfLastChange++;
                   }
               }
               if( ageOfLastChange>5 ){
                   shakeIncludeVector(bp);
               }
               //backtesting(bp);
           }//while
           System.out.format("Ended with age=%d\n",ageOfLastChange);
           System.out.format("Final maximum gain=%.2f, %s\n",currentGain, Arrays.toString(mx.includeVector));
       }
   */
    class CandidateRecord {
        int id;
        double gain;
        int buyThreshold;
        int sellThreshold;

        public CandidateRecord(int _id, double _gain, int _buyThreshold, int _sellThreshold) {
            this.id = _id;
            this.gain = _gain;
            this.buyThreshold = _buyThreshold;
            this.sellThreshold = _sellThreshold;
        }
    }

    class OptimumVector {
        double gain;
        int buyThreshold;
        int sellThreshold;
        char[] vector;

        public OptimumVector(double _gain, int _buyThreshold, int _sellThreshold, char[] _vector, int _num) {
            gain = _gain;
            buyThreshold = _buyThreshold;
            sellThreshold = _sellThreshold;
            vector = new char[_num];
            System.arraycopy(_vector, 0, vector, 0, _num);
        }

        public String format() {
            List<String> items = new ArrayList<>();
            for (int i = 0; i < numModels; i++) {
                if( vector[i] == 'B' ) items.add( modelName[i] );
                if( vector[i] == 'S' ) items.add( modelName[i] );
            }
            return String.format("VECTOR { gain: %.2f, buyThr: %d, sellThr: %d, models: [%s], vector:[%s]\n",
                    gain, buyThreshold, sellThreshold, Arrays.toString(items.toArray()), Arrays.toString(vector) );
        }
    }

    public void runOptimizer2(BacktestingProcessor bp) {
        List<OptimumVector> optimumVectors = new ArrayList<>();
        int epoch = 0;
        int ageOfLastChange = 0;
        double maxGain = 0;
        double currentGain = 0;
        while (continueOptimization() && ageOfLastChange < 20) {
            recalculateSignals(includeVector, buyThreshold, sellThreshold);
            Document result = bp.backtesting(signals[0], signals[1]);
            currentGain = result.getDouble("totalGain");
            System.out.format("Epoch=%d, gain=%.2f, vector=[%s]\n", epoch, currentGain, Arrays.toString(includeVector));
            if (epoch++ > 50) break;

            if (epoch == 0) maxGain = currentGain; // initialize maximum
            if (currentGain > maxGain) {
                maxGain = currentGain;
                System.out.format("New maximum found gain=%.2f, (%d,%d) vector=[%s]\n", maxGain, buyThreshold, sellThreshold, Arrays.toString(includeVector));
                optimumVectors.add( new OptimumVector(currentGain,buyThreshold,sellThreshold,includeVector,numModels) );
            }

            CandidateRecord buyCandidate = findNextBuyCandidate(bp);
            CandidateRecord sellCandidate = findNextSellCandidate(bp);
            if ( (buyCandidate.gain < maxGain && sellCandidate.gain < maxGain) ) {
                // shake model - remove randomly some items
                shakeIncludeVector(bp);
                ageOfLastChange = 0;
            } else if (buyCandidate.gain > sellCandidate.gain && buyCandidate.gain>maxGain) {
                includeVector[buyCandidate.id] = 'B';
                buyThreshold = buyCandidate.buyThreshold;
                sellThreshold = buyCandidate.sellThreshold;
                System.out.format("Adding buy model=%d (%d,%d), gain=%.2f\n", buyCandidate.id, buyCandidate.buyThreshold, buyCandidate.sellThreshold, buyCandidate.gain);
                ageOfLastChange = 0;
            } else  if (sellCandidate.gain > maxGain) {
                includeVector[sellCandidate.id] = 'S';
                buyThreshold = sellCandidate.buyThreshold;
                sellThreshold = sellCandidate.sellThreshold;
                System.out.format("Adding sell model=%d (%d,%d), gain=%.2f\n", sellCandidate.id, sellCandidate.buyThreshold, sellCandidate.sellThreshold, sellCandidate.gain);
                ageOfLastChange = 0;
            }else{
                if( ageOfLastChange>5 ){
                    includeVector[buyCandidate.id] = 'B';
                    includeVector[sellCandidate.id] = 'S';
                }

            }
            ageOfLastChange = ageOfLastChange + 1;
        }
        System.out.format("Ended with age=%d\n", ageOfLastChange);
        System.out.format("Final maximum gain=%.2f, %s\n", currentGain, Arrays.toString(includeVector));
        System.out.println("\nOPTIMUM VECTORS:\n");
        for (OptimumVector optimumVector : optimumVectors) {
            System.out.println( optimumVector.format() );
        }
    }

    public void shakeIncludeVector(BacktestingProcessor bp) {
        // shake the model
        List<Integer> buys = new ArrayList<>();
        List<Integer> sells = new ArrayList<>();
        int k = 0;
        for (char c : includeVector) {
            if (c == 'B') buys.add(k);
            if (c == 'S') sells.add(k);
            k++;
        }
        int eliminateBuy = (int) Math.floor(Math.random() * buys.size());
        int eliminateSell = (int) Math.floor(Math.random() * sells.size());

        if (buys.size() > 1 && sells.size() <= 1) {
            includeVector[buys.get(eliminateBuy)] = '-';
            System.out.format("Removing buy signal at index: %d\n", buys.get(eliminateBuy));
        } else if (buys.size() <= 1 && sells.size() > 1) {
            includeVector[sells.get(eliminateSell)] = '-';
            System.out.format("Removing sell signal at index: %d\n", sells.get(eliminateSell));
        } else if (buys.size() > 1 && sells.size() > 1) {
            if (Math.random() >= 0.5) {
                includeVector[buys.get(eliminateBuy)] = '-';
                System.out.format("Removing buy signal at index: %d\n", buys.get(eliminateBuy));
            } else {
                includeVector[sells.get(eliminateSell)] = '-';
                System.out.format("Removing sell signal at index: %d\n", sells.get(eliminateSell));
            }
        }
        boolean first = true;
        double _maxGain = 0;
        int numB = count('B',includeVector);
        int numS = count('S',includeVector);
        for (int i = buyThreshold-1; i < numB; i++) {
            for (int j = sellThreshold-1; j < numS; j++) {
                recalculateSignals(includeVector, i, j);
                Document result = bp.backtesting(signals[0], signals[1]);
                double totalGain = result.getDouble("totalGain");
                if (totalGain > _maxGain || first) {
                    _maxGain = totalGain;
                    buyThreshold = i > 0 ? i : 1;
                    sellThreshold = j > 0 ? j : 1;
                    first = false;
                }
            }
        }

    }

    private int count( char A, char[] vector ){
        int count = 0;
        for ( char c : vector ){
            if ( A == c ) count++;
        }
        return count;
    }

    private CandidateRecord findNextBuyCandidate(BacktestingProcessor bp) {
        double _maxGain = 0;
        int nextBuyCandidate = -1;
        int _buyThreshold = buyThreshold;
        int _sellThreshold = sellThreshold;

        for (int mBuy = 0; mBuy < numModels; mBuy++) {
            if (includeVector[mBuy] != '-') continue;
            char[] newVector = new char[numModels];
            System.arraycopy(includeVector, 0, newVector, 0, numModels);
            newVector[mBuy] = 'B';

            int numB = count('B',newVector);
            int numS = count('S',newVector);
            for (int i = _buyThreshold-1; i < numB; i++) {
                for (int j = _sellThreshold-1; j < numS; j++) {
                    recalculateSignals(newVector, i, j);
                    Document result = bp.backtesting(signals[0], signals[1]);
                    double totalGain = result.getDouble("totalGain");
                    if (totalGain > _maxGain || nextBuyCandidate == -1) {
                        _maxGain = totalGain;
                        nextBuyCandidate = mBuy;
                        _buyThreshold = i > 0 ? i : 1;
                        _sellThreshold = j > 0 ? j : 1;
                    }
                }
            }
        }
        return new CandidateRecord(nextBuyCandidate, _maxGain, _buyThreshold, _sellThreshold);
    }

    private CandidateRecord findNextSellCandidate(BacktestingProcessor bp) {
        double _maxGain = 0;
        int nextSellCandidate = -1;
        int _buyThreshold = buyThreshold;
        int _sellThreshold = sellThreshold;

        for (int mSell =0; mSell < numModels; mSell++) {
            if (includeVector[mSell] != '-') continue;
            char[] newVector = new char[numModels];
            System.arraycopy(includeVector, 0, newVector, 0, numModels);
            newVector[mSell] = 'S';

            int numB = count('B',newVector);
            int numS = count('S',newVector);
            for (int i = _buyThreshold-1; i < numB; i++) {
                for (int j = _sellThreshold-1; j < numS; j++) {
                    recalculateSignals(newVector, i, j);
                    Document result = bp.backtesting(signals[0], signals[1]);
                    double totalGain = result.getDouble("totalGain");
                    if (totalGain > _maxGain || nextSellCandidate == -1) {
                        _maxGain = totalGain;
                        nextSellCandidate = mSell;
                        _buyThreshold = i > 0 ? i : 1;
                        _sellThreshold = j > 0 ? j : 1;
                    }
                }
            }
        }
        return new CandidateRecord(nextSellCandidate, _maxGain, _buyThreshold, _sellThreshold);
    }
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
