package trainingTasks;

import environment.Utilities;
import neural.DeepLayer;

import java.io.BufferedReader;
import java.io.DataInput;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TrainingProcessor {

    String[] dates;
    public double[][] inputVector;
    String[] inputKPIs;
    public double[] buySignal;
    public double[] sellSignal;
    double[] zigZag;
    double[] price;
    int inputSize;
    int samples;
    public double[] avg;
    public double[] stdev;

    public void loadDataSet(String dataFile, String[] filters, String[] _inputKPIs) {
        inputKPIs = _inputKPIs;
        List<String> days = new ArrayList<>();
        List<Integer> inputColumns = new ArrayList<>();
        List<double[]> records = new ArrayList<>();
        List<Integer> priceColumns = new ArrayList<>();
        List<double[]> pricing = new ArrayList<>();
        String[] priceVector = {"buySignal8","sellSignal8","pf8","zigZag","close","adjClose"};
        String line;

        // Use try-with-resources to ensure the BufferedReader is closed automatically
        try (BufferedReader br = new BufferedReader(new FileReader(dataFile))) {

            int nLine=0;
            while ((line = br.readLine()) != null) {
                if( nLine++==0 ){
                    String[] values = line.split(","); //DEFAULT_DELIMITER + "(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
                    for( String kpi : inputKPIs ){
                        int col=0;
                        for( String v : values ) {
                            if (kpi.contentEquals(v)) {
                                inputColumns.add(col);
                            }
                            col++;
                        }
                    }
                    System.out.println( "Input columns vector = "+Arrays.toString(inputColumns.toArray()) );
                    for( String y : priceVector ){
                        int col=0;
                        for( String v : values ) {
                            if (y.contentEquals(v)) {
                                priceColumns.add(col);
                            }
                            col++;
                        }
                    }
                    inputSize = inputColumns.size();
                    System.out.println( "Output columns vector = "+Arrays.toString(priceColumns.toArray()) );
                    if(inputSize==0) return;
                    continue;
                }
                String[] values = line.split(","); //DEFAULT_DELIMITER + "(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
                String year = values[0].substring(0, 4);
                boolean includeFlag=false;
                for( String yearFilter : filters ){
                    if( yearFilter.equals(year) ){
                        includeFlag=true;
                        break;
                    }
                }
                if( includeFlag ){
                    // Clean up any residual quotes from the split process
                    days.add( values[0] );
                    int pos = 0;
                    double[] v = new double[inputSize];
                    for (int i : inputColumns ) {
                        v[pos++] = Double.parseDouble( values[i] );
                    }
                    records.add(v);
                    double[] y = new double[priceColumns.size()];
                    pos=0;
                    for (int i : priceColumns ) {
                        y[pos++] = Double.parseDouble( values[i] );
                    }
                    pricing.add(y);

                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        //System.out.println("records=" + records.size());
        //System.out.println("cols=" + records.get(0).length);

        samples = records.size();
        inputVector = new double[samples][inputSize];
        dates = new String[samples];
        buySignal = new double[samples];
        sellSignal = new double[samples];
        zigZag = new double[samples];
        price = new double[samples];

        for( int k=0; k<samples; k++ ) {
            dates[k] = days.get(k);
            for( int i=0; i < inputSize; i++) {
                inputVector[k][i] = records.get(k)[i];
            }
            buySignal[k] = pricing.get(k)[0];
            sellSignal[k] = pricing.get(k)[1];
            zigZag[k] = pricing.get(k)[3];
            price[k] = pricing.get(k)[5];

            if( k%50==0) System.out.println( k + "["+dates[k]+"]-> " + Arrays.toString(inputVector[k]) );
        }
        normalizeInputs("closeMA200xo",4.74292099,14.26543601);
        normalizeInputs("closeMA50xo",1.27341681,6.92262897);
        normalizeInputs("cmf",0.03265432,0.20682017);
        normalizeInputs("macd",0.52668275,3.23695993);
        normalizeInputs("macdSignal",0.51746509,3.04182862);
        normalizeInputs("atrDaily",3.88681522,6.09221636);
        normalizeInputs("atr",3.88247562,5.75621837);
        normalizeInputs("atrPct",2.3035259,3.42048145);
        normalizeInputs("mfi",52.80884259,76.37326945);
        normalizeInputs("pvo",-0.69947629,8.27405592);
        normalizeInputs("obv",3.74872051,24.64325854);
        normalizeInputs("willR",44.20454974,69.46943854);
        normalizeInputs("kcLPct",-4.90150646,7.7722257);
        normalizeInputs("kcMPct",0.30185095,1.48703185);
        normalizeInputs("kcUPct",4.29372565,7.92920582);
        normalizeInputs("macdv",19.9256551,82.23265115);
        normalizeInputs("macdvSignal",19.73303331,77.78983534);
        normalizeInputs("mPhase",49.90514991,78.88413791);
        normalizeInputs("mDir",0.52910053,100.03080758);
    }


    public void calculateKPIStats() {
        if( inputVector==null ) return;

        double[] sumX = new double[inputSize];
        double[] sumX2 = new double[inputSize];
        double[] nX = new double[inputSize];

        for( int i=0; i<samples; i++ ){
            for( int j=0; j<inputSize; j++ ){
                sumX[j] += inputVector[i][j];
                sumX2[j] += Math.pow(inputVector[i][j],2);
                nX[j] += 1;
            }
        }

        avg = new double[inputSize];
        stdev = new double[inputSize];

        for( int i=0; i<inputSize; i++ ){
            avg[i] = nX[i]>0 ? sumX[i]/nX[i] : 0;
            stdev[i] = nX[i]>1 ? Math.sqrt( (sumX2[i]+Math.pow(sumX[i],2)/nX[i]) / (nX[i]-1) ) : 0;
            System.out.format("Standardizing column %d:  avg=%.4f, stdev=%.4f\n",i,avg[i],stdev[i]);
        }
    }

    public void normalizeInputs(String kpi, double mu, double sigma) {
        for( int i=0; i< inputSize; i++ ){
            if( inputKPIs[i].contentEquals(kpi) ) {
                for( int d = 0; d < inputVector.length; d++) {
                    inputVector[d][i] = (inputVector[d][i]-mu)/sigma;
                }
                break;
            }
        }
    }

    public void writeTrainingSet(String tsFile) {
        StringBuilder sb = new StringBuilder();
        for( int i=0; i<samples; i++ ){
            sb.append( String.format("%10s, %10.2f, %10.2f, %3.1f, %3.1f",
                    dates[i], price[i], zigZag[i], buySignal[i], sellSignal[i] ) );
            for( double x : inputVector[i] ) {
                sb.append( String.format(", %.5f", x ) );
            }
            sb.append("\n");
        }
        Utilities.writeFile(tsFile, sb);
    }

    public void writePredictions(DeepLayer nn1, double[] buySignal, DeepLayer nn2, double[] sellSignal, String outFile) {

        StringBuilder sb = new StringBuilder();
        int TT1=0;
        int FF1=0;
        int TF1=0;
        int FT1=0;
        int TT2=0;
        int FF2=0;
        int TF2=0;
        int FT2=0;

        for( int i=0; i<samples; i++ ){
            double[] signal1 = nn1.feedForward(inputVector[i]);
            double[] signal2 = nn2.feedForward(inputVector[i]);
            sb.append( String.format("%10s, %10.2f, %10.2f, %3.1f, %3.1f, %3.1f, %3.1f",
                    dates[i], price[i], zigZag[i], buySignal[i], sellSignal[i], signal1[0], signal2[0] ) );
            sb.append("\n");

            boolean exp1 = signal1[0]>0.8;
            boolean act1 = buySignal[i]>0.8;
            if( exp1 && act1 ) TT1++;
            if( !exp1 && !act1 ) FF1++;
            if( !exp1 && act1 ) FT1++;
            if( exp1 && !act1 ) TF1++;

            boolean exp2 = signal2[0]>0.8;
            boolean act2 = sellSignal[i]>0.8;
            if( exp2 && act2 ) TT2++;
            if( !exp2 && !act2 ) FF2++;
            if( !exp2 && act2 ) FT2++;
            if( exp2 && !act2 ) TF2++;

        }
        double recall1 = (TT1+TF1)>0 ? TT1*100.0/(TT1+TF1) : 0;
        double precision1 = (TT1+FT1)>0 ? TT1*100.0/(TT1+FT1) : 0;
        double recall2 = (TT2+TF2)>0 ? TT2*100.0/(TT2+TF2) : 0;
        double precision2 = (TT2+FT2)>0 ? TT2*100.0/(TT2+FT2) : 0;
        String buf1 = String.format("\nbuy.side=[ %3d, %3d, %3d, %3d], precision.buy=%.3f, recall.buy=%.3f",TT1, TF1, FT1, FF1, precision1, recall1);
        System.out.println( buf1 );
        String buf2 = String.format("\nsell.side=[ %3d, %3d, %3d, %3d], precision.sell=%.3f, recall.sell=%.3f",TT2, TF2, FT2, FF2, precision2, recall2);
        System.out.println( buf2 );
        sb.append(buf1).append(buf2);
        Utilities.writeFile(outFile, sb);


    }


    public static void train(DeepLayer network, double[] y, double[][] x, double learningRate, int iterationsNum, String outFile){
        int tsSize = y.length;
        network.LEARNING_RATE = learningRate;
        int milestone1 = iterationsNum/4;
        int milestone2 = iterationsNum/2;
        int milestone3 = (3*iterationsNum)/4;
        double[] entropy = new double[ iterationsNum/500 ];

        List<Integer> Signal1 = new ArrayList<Integer>();
        List<Integer> Signal0 = new ArrayList<Integer>();
        StringBuilder sb = new StringBuilder();

        for( int i=0; i< tsSize; i++ ) {
            if( y[i]>0.8 ) Signal1.add(i); else Signal0.add(i);
        }

        System.out.format("Training signal(1)=%d\n",Signal1.size());
        System.out.format("Training signal(0)=%d\n",Signal0.size());

        if(Signal1.isEmpty() || Signal0.isEmpty()) {
            System.out.println("Aborting...");
            return;
        }

        for (int epoch = 0; epoch < iterationsNum; epoch++) {
            if( epoch==milestone1 ) {
                network.LEARNING_RATE = learningRate*2;
                System.out.format("\nChanging eta=%.3f\n",network.LEARNING_RATE);
            }
            if( epoch==milestone2 ) {
                network.LEARNING_RATE = learningRate;
                System.out.format("\nChanging eta=%.3f\n",network.LEARNING_RATE);
            }
            if( epoch==milestone3 ) {
                network.LEARNING_RATE = learningRate/2;
                System.out.format("\nChanging eta=%.3f\n",network.LEARNING_RATE);
            }

            if( epoch<milestone2 && epoch%250 == 0 ){
                for( int s1 : Signal1 ) {
                    network.train(x[s1], y[s1]);
                }
            }

            int patternIndex = Signal0.get( (int) Math.floor(Math.random()*Signal0.size()) );
            network.train(x[patternIndex], y[patternIndex]);

            patternIndex = Signal1.get( (int) Math.floor(Math.random()*Signal1.size()) );
            network.train(x[patternIndex], y[patternIndex]);

            // Print error every 5000 epochs
            if (epoch % 500 == 0) {
                double totalError = 0;
                int TT=0;
                int FF=0;
                int TF=0;
                int FT=0;
                for (int i = 0; i < x.length; i++) {
                    double[] output = network.feedForward(x[i]);
                    totalError += network.calculateMSE(output, y);
                    boolean exp = output[0]>0.8;
                    boolean act = y[i]>0.8;

                    if( exp && act ) TT++;
                    if( !exp && !act ) FF++;
                    if( !exp && act ) FT++;
                    if( exp && !act ) TF++;
                }
                double recall = (TT+TF)>0 ? TT*100.0/(TT+TF) : 0;
                double precision = (TT+FT)>0 ? TT*100.0/(TT+FT) : 0;

                int phase = epoch/500;
                if( phase<entropy.length ) entropy[phase] = totalError;
                String buf = String.format("Epoch %d: Average MSE = %.6f, TT=%d, TF=%d, FT=%d, FF=%d, Precision=%.3f, Recall=%.3f\n",
                        epoch, totalError, TT, TF, FT, FF, precision, recall);
                sb.append( buf );
                System.out.print(buf);
            }//if


            //write predictions
            for (int i = 0; i < x.length; i++) {
                double[] output = network.feedForward(x[i]);
            }

        }

    }

}
