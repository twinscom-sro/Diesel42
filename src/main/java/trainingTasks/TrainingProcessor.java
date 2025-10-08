package trainingTasks;

import java.io.BufferedReader;
import java.io.DataInput;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TrainingProcessor {

    String[] dates;
    double[][] inputVector;
    double[] buySignal;
    double[] sellSignal;
    double[] zigZag;
    double[] price;
    int inputSize;
    public double[] avg;
    public double[] stdev;

    public void loadDataSet(String dataFile, String[] filters, String[] inputKPIs) {
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
                    pricing.add(v);

                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        //System.out.println("records=" + records.size());
        //System.out.println("cols=" + records.get(0).length);

        int samples = records.size();
        int inputSize = inputColumns.size();
        inputVector = new double[samples][inputSize];
        dates = new String[samples];
        buySignal = new double[samples];
        sellSignal = new double[samples];
        zigZag = new double[samples];
        price = new double[samples];

        int k=0;
        for( double[] v : records) {
            dates[k] = days.get(k);
            for( int i=0; i < inputSize; i++) {
                inputVector[k][i] = v[i];
            }
            buySignal[k] = pricing.get(k)[0];
            sellSignal[k] = pricing.get(k)[1];
            zigZag[k] = pricing.get(k)[3];
            price[k] = pricing.get(k)[5];

            if( k%50==0) System.out.println( k + "["+dates[k]+"]-> " + Arrays.toString(v) );
            k++;
        }
    }


    public void calculateKPIStats() {
        if( inputVector==null ) return;

        double[] sumX = new double[inputSize];
        double[] sumX2 = new double[inputSize];
        double[] nX = new double[inputSize];

        for( int i=0; i<inputVector.length; i++ ){
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
}
