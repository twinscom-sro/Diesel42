package vectorTasks;

import java.util.Arrays;

public class py {

    public static void print( String label, double[] a) {
		if( label==null) System.out.print("[ ");
		if( label!=null) System.out.print(label + " = [ ");
		for( int j=0; j<a.length; j++ ) { 
			System.out.format(" %10.3f",a[j]);
		}
		System.out.println(" ]");
    }

    public static void print( String label, float[] a) {
		if( label==null) System.out.print("[ ");
		if( label!=null) System.out.print(label + " = [ ");
		for( int j=0; j<a.length; j++ ) { 
			System.out.format(" %10.3f",a[j]);
		}
		System.out.println(" ]");
    }
    

    public static void print( String label, double[][] x) {
		if( label!=null) System.out.println(label + ":");
		for( int i=0; i<x.length; i++ ) { 
			System.out.print(" | ");
			for( int j=0; j<x[i].length; j++ ) { 
				System.out.format(" %10.3f",x[i][j]);
			}
			System.out.println(" |");
		}
    }

    public static String toString( double[] x) {
    	String buf="[";
		for( int i=0; i<x.length; i++ ) {
			buf += String.format(" %10.3f",x[i]);
		}
		buf += " ]";
		return buf;
    }

    
    public static double[] copy(double[] a, double[] b) {
    	for( int i=0; i<a.length; i++ ) {
    		a[i] = i<b.length ? b[i] : 0.0;
    	}
    	return a;
    }

    public static double[] copy(double[] a, int length) {
    	double[] x = new double[length];
    	for( int i=0; i<length; i++ ) {
    		x[i] = i<a.length ? a[i] : 0.0;
    	}
    	return x;
    }

    public static double[] copyOffset(double[] a, int length, int offset) {
    	double[] x = new double[length];
    	for( int i=0; i<length; i++ ) {
    		x[i] = 0.0;
    		int index = i-offset;
    		if( index>=0 && index<a.length ) x[i]=a[index];
    	}
    	return x;
    }
    
    public static double[] zeros( int length) {
    	double[] x = new double[length];
    	for( int i=0; i<length; i++ ) x[i]=0.0;
    	return x;
    }
    
    public static double[] multiply(double[] a, double[] b) {
    	double[] x = new double[a.length];
    	for( int i=0; i<a.length; i++ ) {
    		x[i] = i<b.length ? a[i]*b[i] : 0.0;
    	}
    	return x;
    }

    public static double[] multiply(double[] a, double b) {
    	double[] x = new double[a.length];
    	for( int i=0; i<a.length; i++ ) {
    		x[i] = a[i]*b;
    	}
    	return x;
    }

	public static double[] add(double[] a, double b) {
    	double[] x = new double[a.length];
    	for( int i=0; i<a.length; i++ ) {
    		x[i] = a[i]+b;
    	}
    	return x;
	}
	
    public static double sum(double[] a) {
    	double sumA=0.0;
    	for( int i=0; i<a.length; i++ ) {
    		sumA += a[i];
    	}
    	return sumA;
    }

    public static double max(double[] a) {
    	double maxA=a[0];
    	for( int i=1; i<a.length; i++ ) {
    		if( a[i]>maxA ) maxA=a[i];
    	}
    	return maxA;
    }

    public static float max(float[] a) {
    	float maxA=a[0];
    	for( int i=1; i<a.length; i++ ) {
    		if( a[i]>maxA ) maxA=a[i];
    	}
    	return maxA;
    }

    
    public static double min(double[] a) {
    	double minA=a[0];
    	for( int i=1; i<a.length; i++ ) {
    		if( a[i]<minA ) minA=a[i];
    	}
    	return minA;
    }

	public static int min(int[] a) {
		int minA=a[0];
		for( int i=1; i<a.length; i++ ) {
			if( a[i]<minA ) minA=a[i];
		}
		return minA;
	}

	public static double sigmoid(double x) {
        return( 1.0 / (1 + Math.exp(-x)) );
    }

    
    public static double[] randomVector(int length, double yMin, double yMax) {
    	double[] x = new double[length];
    	for( int i=0; i<length; i++ ) {
    		x[i] = Math.random()*(yMax-yMin)+yMin;
    	}
    	return x;
    }
    
    public static double randomValue( double yMin, double yMax) {
    	return( Math.random()*(yMax-yMin)+yMin );
    }

	public static String printStats(double[] y) {
		
		int n=y.length;
		double[] x = new double[n];
		for( int i=0; i<n; i++ ) {
			x[i]=y[i];
		}
		Arrays.sort(x);
		/*for( int i=0; i<n; i++ ) {
			System.out.printf("%f,",x[i]);
		}
		System.out.println();*/
		int n2 = n/2;
		int n4 = n/4;
		double fq1=(n-1)/4.0;
		double fq3=(n-1)*3.0/4.0;
		int iq1 = (int)Math.floor(fq1);
		int iq3 = (int)Math.floor((n-1)*3.0/4.0);
		double factor1 = fq1-iq1;
		double factor2 = fq3-iq3;
		if( factor2>0 ) iq3++;

		//System.out.printf("f=%f %f, %d : %d \n",factor1, factor2, iq1,iq3);
		//System.out.printf("%f %f  || %f %f\n",x[iq1],x[iq1-1],x[iq3],x[iq3-1]);	
		
		double q2 = (n%2!=0) ? x[n/2] : ( x[n/2]+x[n/2-1] )/2; 
		double q1 = factor1==0 ? x[iq1] : x[iq1]+((x[iq1+1]-x[iq1])*factor1); 
		double q3 = factor2==0 ? x[iq3] : x[iq3-1]+((x[iq3]-x[iq3-1])*factor2); 
		
		double avg=0.0;
		double var=0.0;
		for( int i=0; i<n; i++ ) {
			avg += x[i];
		}
		avg = avg / n;
		for( int i=0; i<n; i++ ) {
			var += Math.abs(x[i]-avg);
		}
		var = var / n;
		
		String out = String.format(
				"min=%3.2f, q1=%3.2f, q2=%3.2f, q3=%3.2f, max=%3.2f, avg=%3.2f, var=%3.2f", 
				x[0], q1, q2, q3, x[n-1], avg, var );
		return out;
	}

	static final double[] test= {
			1.91412735689137,
			8.7921822723789,
			1.12557699483844	,
			7.66582282876733	,
			4.79509481401462	,
			1.01030696156013	,
			3.7070821015365	,
			1.55909083212571	,
			3.36152616256178	,
			5.60537502035208	,
			3.29315230246738	,
			3.00997449970395	,
			8.79569746153985	,
			3.27527919547854	,
			5.8435077403058	,
			2.97940210185955	,
			3.60684243421862	,
			8.82621842087885	,
			0.506810561777944	,
			2.04377540322873	,
			6.17924589039795	,
			6.29511896792313	,
			0.318017782408491	,
			0.728101480668621	,
			3.22156664671015	,
			1.56520488847259	,
			4.52096175985322	,
			2.22751064436898	,
			4.42312344247289	,
			9.01065105058509	,
			7.87425777580962	,
			9.84315311661637	,
			5.58062340522055	,
			8.59793327795239	,
			4.78730333787361	,
			8.52629091322642	,
			8.55327147241272	,
			3.02517916346847	,
			3.00841697125177	,
			6.3676273852228	,
			1.23772907222634	,
			4.62699668861397	,
			5.91454207852107	,
			0.736618706084489	,
			8.69856257098726	,
			0.357931664656266	,
			3.17317991759102	,
			9.16004686733695	,
			3.44877521179045	,
			6.70245559379818	,
			3.11307470001873	,
			7.75783655288544	,
			9.23960451848567	,
			5.09686953499102	,
			3.98820318282663	,
			3.62890008572513	,
			9.3529310930124	,
			0.240375083677486	,
			0.398864092504656	,
			2.53805806212463	,
			1.56886168613185	,
			1.46666531167091	,
			7.79918591904889	,
			6.9725660627735	};//,
			//5.02869553148495,
			//4.87775624973265 };//,
			//4.08834734790871 };
	

	public static void main(String[] args) {
		double[] y = {3.2, 1.2, 9.7, 6.2, 4.5, 3.1, 1.1, 9.1, 6.1, 4.1 };
		double[] z = { -0.64, -1.56, 3.12, 0.17, -2.07, -1.90, 1.77, -1.34, -6.35, -5.39, -2.41, 0.92, 2.28, 0.72, -4.11, 3.14, 3.63, 9.44, -0.58, 0.06, 2.03, 3.07, -2.98, -1.96, 3.77, 10.09, 1.72, -0.44, 0.35, 2.79, 3.97, 4.96, 0.80, 1.12, 1.60, 4.66, 0.25, 1.10, -0.62, 3.58, 9.99, 1.44, -12.47, 16.99, 13.76, 0.45, 16.64, -13.64, 8.06, -0.98, 6.42, 13.91, 7.30, 8.00, -7.03, 3.13, -5.26, 2.98 };		
		
		System.out.println( printStats(z) );
	}

	public static double[] subset(double[] y, int n) {
		if( n==0 ) n=1;
		double[] x = new double[n];
		for( int i=0; i<n; i++ ) x[i]=y[i];
		return x;
	}

	public static double[] normalize(double[] a) {
		int n = a.length;
		if( n<=2 ) return a;

		double sum=0.0;
		for( int i=0; i<n; i++ ) sum += a[i];
		double mean = sum/n;
		
		sum=0.0;
		for( int i=0; i<n; i++ ) sum += (a[i]-mean)*(a[i]-mean);		
		double stdev = Math.sqrt( sum / (n-1) );

    	double[] x = new double[a.length];
    	for( int i=0; i<a.length; i++ ) {
    		x[i] = (a[i]-mean)/stdev;
    	}
    	return x;
	}

	public static double[] logNormalize(double[] a) {
		int n = a.length;
		if( n<=2 ) return a;

		double sum=0.0;
		for( int i=0; i<n; i++ ) sum += Math.log(a[i]);
		double mean = sum/n;
		
		sum=0.0;
		for( int i=0; i<n; i++ ) sum += (Math.log(a[i])-mean)*(Math.log(a[i])-mean);		
		double stdev = Math.sqrt( sum / (n-1) );

    	double[] x = new double[a.length];
    	for( int i=0; i<a.length; i++ ) {
    		x[i] = (Math.log(a[i])-mean)/stdev;
    	}
    	return x;
	}

    
}