package vectorTasks;

public class Series {
	
	public float[] x;
	
/*	public Series( DataFrame df, int offset, int length, int mode ){
		x = df.subset(offset, length, mode);
	}

	public Series( DataFrame df, int mode ){
		x = df.subset(0, df.numPoints, mode);
	}


	public Series(DataFrame df, String filter, int mode) {
		x = df.FilterByYear(filter,mode).x;	
	}
*/
public Series(int size ){
	x = new float[size];
	for( int i=0; i<size; i++ ) x[i]=0;
}

	public Series(Series y ){
		x = y.x.clone();
	}

	public Series(double[] y) {
		x = new float[y.length];
		for( int i=0; i<y.length; i++ ) x[i]=(float)y[i];
	}

	public float get( int i ) { return ( i<0 || i>=x.length )? 0 : ( Double.isNaN(x[i]) || Double.isInfinite(x[i]) ? 0 : x[i] ); }
	public void set( int i, float value ) { if( i>=0 && i<x.length ) x[i] = value; }
	public int length() { return x.length; }

	
	public void print(String title) {
		System.out.println( String.format("%10s", title) );
		for( int i=0; i<x.length; i++ )
			System.out.println( String.format("%10.2f", x[i]) );
		
	}

	public void printXY(String title1, String title2, Series y) {
		System.out.println( String.format("%12s %12s", title1, title2) );
		for( int i=0; i<x.length; i++ )
			System.out.format("%12.4f %12.4f\n", x[i], y.get(i) );
		
	}
	
	public String sprintXY(String title1, String title2, Series y) {
		StringBuilder out = new StringBuilder();
		out.append( String.format("%12s %12s\n", title1, title2) );
		for( int i=0; i<x.length; i++ )
			out.append( String.format("%12.4f %12.4f\n", x[i], y.get(i) ) );
		return out.toString();
	}

	public void printAttr(String attr, float value ) {
		System.out.format("%15s = %15.3f\n", attr, value );		
	}

	public float avg() {
		double sum=0.0;
		for( int i=0; i<x.length; i++ ) sum += x[i];
		return (float) sum/x.length;
	}
	
	public float min() {
		float y=x[0];
		for( int i=1; i<x.length; i++ ) {
			if( x[i]<y ) y=x[i];
		}
		return y;
	}

	public float max() {
		float y=x[0];
		for( int i=1; i<x.length; i++ ) {
			if( x[i]>y ) y=x[i];
		}
		return y;
	}

	public float std() {
		float xAvg = avg();
		double sum=0.0;
		for( int i=0; i<x.length; i++ ) sum += (x[i]-xAvg)*(x[i]-xAvg);		
		return (float) Math.sqrt( sum / (num()-1) );
	}

	public float num() {
		return x.length;
	}

	
	public Series shift( int offset ) {
		Series y = new Series( x.length+offset );
		for( int i=0; i<y.num(); i++ ) y.set(i , get(i-offset) );
		return y;
	}

	public Series calculateReturnSeries() {
		Series y = this.shift(-1);
		for( int i=0; i<y.num(); i++ ) y.set(i, x[i]!=0 ? y.get(i)/x[i] : 0 );
		return y;
	}
	
	public Series log( Series z ) {
		Series y = new Series( z );
		for( int i=0; i<y.x.length; i++ ) y.set( i, (float)( z.get(i)>0 ? Math.log(z.get(i)) : 0)  );
		return y;
	}

	public Series calculateLogReturnSeries() {
		return log( calculateReturnSeries() );
	}

	public float calculateCAGR() {
		double year_portion = x.length/365.25;
		double yield = get(x.length-1) / get(0);
		return (float)( Math.pow(yield,(1 / year_portion)) - 1 );
	}

	public Series EMA(double factor) {
		Series y = new Series( x.length );
		double yVal = x[0];
		for( int i=0; i<x.length; i++ ) {
			yVal = yVal*(1-factor) + x[i]*(factor);
			y.set( i , (float)yVal ); 
		}
		return y;
	}

	public Series MA(int days) {
		Series y = new Series( x.length );
		for( int i=0; i<x.length; i++ ) {
			y.set(i, this.average(i-days+1,i) );
		}
		return y;
	}

	private float average(int iStart, int iEnd) {
		float sum=0;
		if( iStart<0 ) iStart=0;
		if( iEnd>=x.length ) iEnd = x.length-1;
		int n=0;
		for( int i=iStart; i<=iEnd; i++,n++ ) sum+=x[i];
		return( n>0 ? sum / n : 0 );
	}

	public Series normalize() {
		float mean = avg();
		float stdev = std();
		Series y = new Series( x.length );
		for( int i=0; i<x.length; i++ ) {
			y.set(i, (x[i]-mean)/stdev );
		}
		return y;
	}

	
	public Series firstDerivative(float factor) {
		Series y = new Series( x.length );
		for( int i=1; i<x.length; i++ ) {
			y.set(i, (x[i]-x[i-1])*factor );
		}
		y.set(0, y.get(1) );
		return y;
	}

	public Series smooth(int days) {
		Series y = new Series( x.length );
		for( int i=0; i<x.length; i++ ) {
			y.set(i, this.average(i-days,i+days) );
		}
		return y;
	}

	public Series subtract(Series diff) {
		Series y = new Series( x.length );
		for( int i=0; i<x.length; i++ ) {
			y.set(i, get(i) - diff.get(i) );
		}
		return y;
	}

	public Series addMult(Series a, float mag) {
		Series y = new Series( x.length );
		for( int i=0; i<x.length; i++ ) {
			y.set(i, get(i) + mag*a.get(i) );
		}
		return y;
	}

	public Series assignCrossovers(Series signal2, int buy, int sell) {
		Series y = new Series( x.length );
		for( int i=1; i<x.length; i++ ) {
			y.set(i, 0);
			if( (x[i-1]<0 && x[i]>=0) && signal2.get(i)>0 ) y.set(i, buy);
			if( (x[i-1]<0 && x[i]>=0) && signal2.get(i)<0 ) y.set(i, sell);
			if( (x[i-1]>0 && x[i]<=0) && signal2.get(i)>0 ) y.set(i, buy);
			if( (x[i-1]>0 && x[i]<=0) && signal2.get(i)<0 ) y.set(i, sell);
		}
		return y;
	}

	public void simTrade(Series action) {
		double sumYield=0.0;
		int numTrades=0;
		
		float position=0;
		int qty=0;
		double gpAvg=0.0;
		double totalGP=1.0;
		for( int i=1; i<x.length; i++ ) {
			if( action.get(i)==1 && qty==0 ) {
				qty = 1;
				position = x[i];
				System.out.format("%2d Buy  %10.2f\n",i,x[i]);
			}
			if( action.get(i)==-1 && qty==1 ) {
				qty = 0;
				numTrades++;
				sumYield += (x[i]-position);
				double gp= (x[i]-position)/position;
				gpAvg += gp;
				totalGP *= (1.0+gp);
				System.out.format("%2d Sell %10.2f, profit = %10.2f, GP=%10.3f\n",i,x[i],(x[i]-position),gp*100.0);
				position = 0;
			}
		}		
		
		System.out.format("Trade Summary:  trades=%2d,  profit = %10.2f, avgGP=%10.3f, sumGP=%10.3f\n",
				numTrades,sumYield,gpAvg*100.0/numTrades,totalGP*100.0);
	}
	
	public void simShortTrade(Series action) {
		double sumYield=0.0;
		int numTrades=0;
		
		float position=0;
		int qty=0;
		double gpAvg=0.0;
		double totalGP=1.0;
		for( int i=1; i<x.length; i++ ) {
			if( action.get(i)==-1 && qty==0 ) {
				qty = 1;
				position = x[i];
				System.out.format("%2d ShortSell  %10.2f\n",i,x[i]);
			}
			if( action.get(i)==1 && qty==1 ) {
				qty = 0;
				numTrades++;
				sumYield += -(x[i]-position);
				double gp= -(x[i]-position)/position;
				gpAvg += gp;
				totalGP *= (1.0+gp);
				System.out.format("%2d ShortCover %10.2f, profit = %10.2f, GP=%10.3f\n",i,x[i],-(x[i]-position),gp*100.0);
				position = 0;
			}
		}		
		
		System.out.format("Trade Summary:  trades=%2d,  profit = %10.2f, avgGP=%10.3f, sumGP=%10.3f\n",
				numTrades,sumYield,gpAvg*100.0/numTrades,totalGP*100.0);
	}

	public Series assignPosition() {
		Series y = new Series( x.length );
		float position=0;
		for( int i=0; i<x.length; i++ ) {
			if( x[i]==1 && position!=1 ) {
				position=1;
			}else if( x[i]==-1 && position!=-1 ) {
				position=-1;
			}
			y.set(i,position);
		}
		return y;
	}

	public Series multiply(float factor) {
		Series y = new Series( x.length );
		for( int i=0; i<x.length; i++ ) {
			y.set(i,x[i]*factor);
		}
		return y;
	}

	public Series divide(Series b) {
		Series y = new Series( x.length );
		for( int i=0; i<x.length; i++ ) {
			float d=b.get(i);
			y.set(i,   d!=0 ? x[i]/d : 0 );
		}
		return y;
	}

	
	public Series add(float factor) {
		Series y = new Series( x.length );
		for( int i=0; i<x.length; i++ ) {
			y.set(i,x[i]+factor);
		}
		return y;
	}
	
	public Series filter(float yMin, float yMax) {
		Series y = new Series( x.length );
		for( int i=0; i<x.length; i++ ) {
			y.set(i, ( x[i]>=yMin && x[i]<=yMax ) ? x[i] : 0 );
		}
		return y;
	}

	public Series smoothExp(double factor) {
		Series y = new Series( x.length );
		for( int i=0; i<x.length; i++ ) {
			y.set(i, x[i] );
			for( int j=1; j<10; j++ ) {
				y.set(i+j, (float) (y.get(i)*(1-factor) + get(i+j)*factor) );
				y.set(i-j, (float) (y.get(i)*(1-factor) + get(i-j)*factor) );
			}
		}
		return y;
	}

	public Series anticipateSignal(float direction,int lead) {
		Series y = new Series( x.length );
		for( int i=0; i<x.length; i++ ) {
			float value=0;
			for( int j=0; j<lead; j++ ) {
				if( get(i+j)>=direction ) value = 1;
				if( get(i+j)<=direction ) value = -1;
			}
			y.set(i,value);
		}
		return y;
	}

	public Series signal(float threshold, int output) {
		Series y = new Series( x.length );
		for( int i=0; i<x.length; i++ ) {
			y.set(i, x[i]>threshold ? output : 0);
		}
		return y;
	}


	public Series calculatePAF(double changePct) {
		Series y = new Series( x.length );
		float curMin=x[0];
		float curMax=x[0];
		int iMin=0;
		int iMax=0;
		float signal=0;
		for( int i=0; i<x.length; i++ ) {
			if( x[i]<curMin ) {
				curMin=x[i];
				iMin=i;
			}
			if( x[i]>curMax ) {
				curMax=x[i];
				iMax=i;
			}
			
			float dFromMax = Math.abs((curMax - x[i])/*/curMax*/);
			float dFromMin = Math.abs((x[i] - curMin)/*/curMin*/);
		
			if( signal==0 ){
				if( dFromMax > changePct ){
					signal = -1;
					for( int j=iMax; j<i; j++ ) y.set(j, signal);
				}
				if( dFromMin > changePct ) {
					signal = 1;
					for( int j=iMin; j<i; j++ ) y.set(j, signal);
				}
			}			
			if( signal==-1 && dFromMin>changePct ){
				signal = 1; // start going up
				for( int j=iMin; j<i; j++ ) y.set(j, signal);
				curMin = x[i];
				curMax = x[i];
				iMin=i;
				iMax=i;
			}			
			if( signal==1 && dFromMax>changePct ){
				signal = -1; // start going down
				for( int j=iMax; j<i; j++ ) y.set(j, signal);
				curMin = x[i];
				curMax = x[i];
				iMin=i;
				iMax=i;
			}				
			
			//System.out.printf("%d %f min: %f %f max %f  %f  signal: %f\n",i,x[i],curMin,dFromMin,curMax,dFromMax, signal);
			y.set(i, signal );
		}
		return y;
	}

	public Series calculatePF(double changePct) {
		if( x.length<=0 ) {
			return new Series( 10 );	
		}
		Series y = new Series( x.length );
		float curMin=x[0];
		float curMax=x[0];
		int iMin=0;
		int iMax=0;
		int direction=0;
		int lastYi=0;
		float lastY=x[0];

		for( int i=1; i<x.length; i++ ) {
			if( x[i]<curMin ) {
				curMin=x[i];
				iMin=i;
			}
			if( x[i]>curMax ) {
				curMax=x[i];
				iMax=i;
			}

			float dFromMax = Math.abs((curMax - x[i])/curMax)*100;
			float dFromMin = Math.abs((x[i] - curMin)/curMin)*100;

			//System.out.printf("%d %f min: %d %f %f max %d %f  %f  signal: %d \n",i,x[i],iMin,curMin,dFromMin,iMax,curMax,dFromMax, direction);
			
			if( direction==0 ){
				if( dFromMax > changePct ){
					// curve down
					//System.out.printf("D %d %d %f\n",0,iMax,lastY);
					for( int j=0; j<iMax; j++ ) y.set(j, lastY);
					direction = -1;
					lastY=curMax;
					lastYi=iMax;
					curMin=x[i];
					iMin=i;
				}else if( dFromMin > changePct ) {
					// curve up
					//System.out.printf("U %d %d %f\n",0,iMin,lastY);
					for( int j=0; j<iMin; j++ ) y.set(j, lastY);
					direction = 1;
					lastY=curMin;
					lastYi=iMin;
					curMax=x[i];
					iMax=i;
				}
			}			

			if( direction==1 && dFromMax > changePct ){
					// curve down
					//System.out.printf("UP %d %d %f\n",lastYi,iMax,lastY);
					for( int j=lastYi; j<iMax; j++ ) y.set(j, lastY);
					lastYi = iMax;
					lastY = curMax;
					direction = -1;
					curMin=x[i];
					iMin=i;
			}else if( direction==-1 && dFromMin > changePct ) {
					// curve up
					//System.out.printf("DOWN %d %d %f\n",lastYi,iMin,lastY);
					for( int j=lastYi; j<iMin; j++ ) y.set(j, lastY);
					lastYi = iMin;
					lastY = curMin;
					direction = 1;
					curMax=x[i];
					iMax=i;
			}

			//System.out.printf(" %f\n",y.get(i) );
			
		}
		for( int j=lastYi; j<x.length; j++ ) y.set(j, direction==1 ? curMin : curMax );
		
		return y;
	}
	
	
	public Series generateBuySignal(float signal, int offsetBefore, int offsetAfter) {
		Series y = new Series( x.length );
		int position=0;
		float price=x[0];
		for( int i=1; i<x.length; i++ ){
			if( x[i]<price ) {
				if( position==0 ){
					//buy
					int pos1 = i-offsetBefore;
					if( pos1<0 ) pos1=0;
					int pos2 = i+offsetAfter;
					if( pos2>=x.length ) pos2=x.length-1;
					for( int j=pos1; j<=pos2; j++ )	y.set(j, signal);
					position = 1;
					price=x[i];
				}else{
					price=x[i];
				}
			}else if( x[i]>price ) {
				if( position==1 ){
					//sell
					position = 0;
					price=x[i];
				}else{
					price=x[i];
				}
			}else{
			}
		}
		return y;
	}

	public Series generateSellSignal(float signal, int offsetBefore, int offsetAfter) {
		Series y = new Series( x.length );
		int position=0;
		float price=x[0];
		for( int i=1; i<x.length; i++ ){
			if( x[i]>price ) {
				if( position==0 ){
					//sell
					int pos1 = i-offsetBefore;
					if( pos1<0 ) pos1=0;
					int pos2 = i+offsetAfter;
					if( pos2>=x.length ) pos2=x.length-1;
					for( int j=pos1; j<=pos2; j++ )	y.set(j, signal);
					position = 1;
					price=x[i];
				}else{
					price=x[i];
				}
			}else if( x[i]<price ) {
				if( position==1 ){
					//close short
					position = 0;
					price=x[i];
				}else{
					price=x[i];
				}
			}else{
			}
		}
		return y;
	}

	public Series calculateZigZag() {
		Series y = new Series( x.length );
		int x1,x2;
		float y1,y2;
		int position=0;
		float price=x[0];
		y1=x[0];
		x1=0;
		for( int i=1; i<x.length; i++ ){
			y.set(i, 0);
			if( x[i]>price ) {
				if( position==0 ){
					//short
					position = -1;
					price=x[i];
					y.drawLine(x1,y1,i,x[i]);
					x1=i;
					y1=price;
				}else{
					// sell & open short
					position = -1;
					price=x[i];
					y.drawLine(x1,y1,i,x[i]);
					x1=i;
					y1=price;
				}
			}else if( x[i]<price ) {
				if( position==0 ){
					//buy
					position = 1;
					price=x[i];
					y.drawLine(x1,y1,i,x[i]);
					x1=i;
					y1=price;
				}else{
					// close short & buy				
					position = 1;
					price=x[i];
					y.drawLine(x1,y1,i,x[i]);
					x1=i;
					y1=price;
				}
			}
		}
		y.drawLine(x1,y1,x.length-1,x[x.length-1]);
		return y;
	}


	public Series calculateZigZag( int minDays ) {
		int MAX_SIZE=300;
		int x1[] = new int[MAX_SIZE];
		float y1[] = new float[MAX_SIZE];
		int n=0;
		Series y = new Series( x.length );
		
		float price=x[0];
		for( int i=1; i<x.length; i++ ){
			if( x[i]!=price ) {
				x1[n]=i;
				y1[n]=x[i];
				price=x[i];
				n++;
			}
		}

		if( n>4 ) {
			int n2=n;
			
			do {
				n=n2;
				n2 = removeTwoPoints( x1, y1, n, minDays);
			}while( n!=n2 );
		}

		for( int i=1; i<n; i++ ){
			y.drawLine(x1[i-1],y1[i-1],x1[i],y1[i]);
		}
		y.drawLine(0,x[0],x1[0],y1[0]);
		y.drawLine(x1[n-1],y1[n-1],x.length-1,x[n-1]);
		
		return y;
	}
	
	
	public Series calculateZigZag( int minDays, float endpoint ) {
		int MAX_SIZE=3000;
		int x1[] = new int[MAX_SIZE];
		float y1[] = new float[MAX_SIZE];
		int n=0;
		Series y = new Series( x.length );
		
		float price=x[0];
		for( int i=1; i<x.length; i++ ){
			if( x[i]!=price ) {
				x1[n]=i;
				y1[n]=x[i];
				price=x[i];
				n++;
			}
		}

		/*for( int i=1; i<n; i++ ){
			if( x[i]-x[i-1] < minDays ) {
				for(int j=i;j<n-1;j++) {
					x1[j]=x1[j+1];
					y1[j]=y1[j+1];
				}
				n--;
			}
		}*/

		if( n>4 ) {
			int n2=n;
			
			do {
				n=n2;
				n2 = removeTwoPoints( x1, y1, n, minDays);
			}while( n!=n2 );
		}

		for( int i=1; i<n; i++ ){
			y.drawLine(x1[i-1],y1[i-1],x1[i],y1[i]);
		}
		y.drawLine(0,x[0],x1[0],y1[0]);
		y.drawLine(x1[n-1],y1[n-1],x.length-1,endpoint);
		
		return y;
	}
	

	boolean removeOnePoint( int[] x, float[] y, int n, int minDays ) {
		if( n<4 ) return false; 
		
		/*int minDist=x[1]-x[0];
		int ndx=1;
		for( int i=2; i<n; i++ ){
			if( (x[i]-x[i-1])<minDist ) {
				minDist =x[i]-x[i-1];
				ndx=i;
			}
		}
		
		if( minDist>=minDays || ndx<=0 || ndx>=(n-3) ) return false;

		int k=ndx;
		
*/		
		boolean change=false;
		
		for( int k=1; k<n-2; k++ ) {
		
		if( (y[k-1]<y[k]) && (y[k-1]<y[k+1]) && (y[k-1]<y[k+2]) && (y[k]<y[k+2]) ) {
			for( int j=k; j<n-2; j++ ) {
				x[j-1]=x[j+1];
				y[j-1]=y[j+1];
			}
			change = true;
			//return true;
		}
		
		if( (y[k-1]>y[k]) && (y[k-1]>y[k+1]) && (y[k-1]>y[k+2]) && (y[k]>y[k+2]) ){
			for( int j=k; j<n-2; j++ ) {
				x[j-1]=x[j+1];
				y[j-1]=y[j+1];
			}
			change = true;
			//return true;
		}
		
		}//for
		
		return change;
	
	}
	
	int removeTwoPoints( int[] x, float[] y, int n, int minDays ) {
		if( n<4 ) return n; 

		for( int k=1; k<n-2; k++ ) {
							
		if( (y[k-1]<y[k]) && (y[k-1]<y[k+1]) && (y[k-1]<y[k+2]) && (y[k]<y[k+2]) 
				&& ( (x[k]-x[k-1])<minDays || (x[k+1]-x[k])<minDays || (x[k+2]-x[k+1])<minDays) ) {
			for( int j=k; j<n-2; j++ ) {
				x[j]=x[j+2];
				y[j]=y[j+2];
			}
			n-=2;
		}
		
		if( (y[k-1]>y[k]) && (y[k-1]>y[k+1]) && (y[k-1]>y[k+2]) && (y[k]>y[k+2]) 
				&& ( (x[k]-x[k-1])<minDays || (x[k+1]-x[k])<minDays || (x[k+2]-x[k+1])<minDays) ){
			for( int j=k; j<n-2; j++ ) {
				x[j]=x[j+2];
				y[j]=y[j+2];
			}
			n-=2;
		}
		
		}//for
		
		return n;

	}
	
	
	private void drawLine(int x1, float y1, int x2, float y2) {
		float alpha=(y2-y1)/(x2-x1);
		for( int i=x1; i<=x2; i++ ) {
			float y = alpha*(i-x1) + y1;
			set(i, y);
		}	
	}

	private void setSlope(int x1, float y1, int x2, float y2){
		float alpha = x1!=x2 ? (y2-y1)/(x2-x1) : 0;
		for( int i=x1; i<=x2; i++ ) {
			set(i, alpha);
		}	
	}

	private void setConstant(int x1, int x2, float c){
		for( int i=x1; i<=x2; i++ ) {
			set(i, c);
		}
	}

		
	public Series calculateSlopes() {
		Series y = new Series( x.length );
		int x1,x2;
		float y1,y2;
		int position=0;
		float price=x[0];
		y1=x[0];
		x1=0;
		for( int i=1; i<x.length; i++ ){
			y.set(i, 0);
			if( x[i]>price ) {
				if( position==0 ){
					//short
					position = -1;
					price=x[i];
					y.setSlope(x1,y1,i,x[i]);
					x1=i;
					y1=price;
				}else{
					// sell & open short
					position = -1;
					price=x[i];
					y.setSlope(x1,y1,i,x[i]);
					x1=i;
					y1=price;
				}
			}else if( x[i]<price ) {
				if( position==0 ){
					//buy
					position = 1;
					price=x[i];
					y.setSlope(x1,y1,i,x[i]);
					x1=i;
					y1=price;
				}else{
					// close short & buy				
					position = 1;
					price=x[i];
					y.setSlope(x1,y1,i,x[i]);
					x1=i;
					y1=price;
				}
			}
		}
		y.setSlope(x1,y1,x.length-1,x[x.length-1]);
		return y;	}

	
	public Series calculateSlopesContinuous(Series yPrice) {
		Series y = new Series( x.length );
		int x1,x2;
		float y1,y2;
		int position=0;
		float price=x[0];
		y1=x[0];
		x1=0;
		for( int i=1; i<x.length; i++ ){
			y.set(i, 0);
			if( x[i]>price ) {
				if( position==0 ){
					//short
					position = -1;
					price=x[i];
					y.setSlope(x1,y1,i,x[i]);
					x1=i;
					y1=price;
				}else{
					// sell & open short
					position = -1;
					price=x[i];
					y.setSlope(x1,y1,i,x[i]);
					x1=i;
					y1=price;
				}
			}else if( x[i]<price ) {
				if( position==0 ){
					//buy
					position = 1;
					price=x[i];
					y.setSlope(x1,y1,i,x[i]);
					x1=i;
					y1=price;
				}else{
					// close short & buy				
					position = 1;
					price=x[i];
					y.setSlope(x1,y1,i,x[i]);
					x1=i;
					y1=price;
				}
			}
		}
		y.setSlope(x1,y1,x.length-1,x[x.length-1]);
		return y;	
	}

	
	
	private void consolidatePF(int[] days, float[] price, int count, int index) {
		printPF( days, price, index );
		if( index>=(count-2) || index<=2 ) return; // if it's last or first, exit
		int d1 = days[index-2];
		int d2 = days[index-2];
		int d3 = days[index];
		int d4 = days[index+1];
		int d5 = days[index+2];
		float p1=price[index-2];
		float p2=price[index-1];
		float p3=price[index];
		float p4=price[index+1];
		float p5=price[index+2];
		
		if( d2>=30 && d3<30 && d4>=30 && p1<p3 & p3<p4 ){
			days[index-1] += days[index];
			days[index] = 0; // eliminate the low.
			price[index+1] = price[index-1];
			printPF( days, price, index );		
		}
	}

	private void printPF(int[] days, float[] price, int index ) {
		for( int i=index-2; i<=index+2; i++ ) {
			if( days[i]!=0 ) System.out.printf(" [%d : %10.2f]", days[i], price[i] );
		}
		System.out.printf("\n" );
	}

	private int getMinDays(int[] days, int count) {
		int min=days[1];
		int index=1;
		for( int i=1; i<count; i++ ) {
			if( days[i]>0 && days[i]<min ) {
				min=days[i];
				index=i;
			}
		}
		System.out.printf("min days found = %d $ pos = %d\n",days[index],index);
		return index;
	}
	

	public double[] subsetArray(int i1, int i2) {
		double[] y = new double[i2-i1+1];
		for( int i=i1; i<=i2; i++ ) {
			y[i-i1] = x[i];
		}
		return y;
	}

	public Series subset(int i1, int i2) {
		double[] y = new double[i2-i1];
		for( int i=i1; i<i2; i++ ) {
			y[i-i1] = x[i];
		}
		return new Series(y);
	}

	
	public static void print(String title, double[] y) {
		System.out.println( String.format("%10s", title) );
		for( int i=0; i<y.length; i++ )
			System.out.println( String.format("%5.4f", y[i]) );		
	}

	public void print(int from, int to ) {
		for( int i=from; i<to; i++ )
			System.out.println( String.format("%5.4f", x[i]) );		
	}

	public Series normalizePlus( Series y2 ) {
		float mean = avg();
		float stdev = std();
		Series y = new Series( x.length );
		for( int i=0; i<x.length; i++ ) {
			y.set(i, (x[i]-mean+2*stdev)/stdev );
			if( y2!=null ) y2.set(i, (y2.x[i]-mean+2*stdev)/stdev );
		}
		return y;
	}	


	public Series normalizeOn( Series yRef ) {
		float mean = yRef.avg();
		float stdev = yRef.std();
		Series y = new Series( x.length );
		for( int i=0; i<x.length; i++ ) {
			y.set(i, (x[i]-mean+2*stdev)/stdev );
		}
		return y;
	}	

	public Series normalizePlus() {
		return normalizePlus(null);
	}

	public void calculateStochastic(Series c, Series h, Series l, int interval) {
		for( int i=0; i<length(); i++ ) {
			float low = l.x[i]; 
			float high = h.x[i]; 
			if( interval>1 ) {
				for( int d=0; d<=interval; d++ ) {
					if( i>=d && l.x[i-d]<low ) low=l.x[i-d];
					if( i>=d && h.x[i-d]>high ) high=h.x[i-d];
				}
			}
			x[i] = (c.x[i]-low)/(high-low);
		}
		
	}

	public void calculateChaikin(Series yClose, Series yHigh, Series yLow, Series yVolume) {
		float n0 = ((yClose.x[0]-yLow.x[0]) - (yHigh.x[0]-yClose.x[0])) / (yHigh.x[0]-yLow.x[0]);
		float m_prior = n0*yVolume.x[0];
		float EMA3 = m_prior;
		float EMA10 = m_prior;
		for( int i=0; i<length(); i++ ) {
			float n = ((yClose.x[i]-yLow.x[i]) - (yHigh.x[i]-yClose.x[i])) / (yHigh.x[i]-yLow.x[i]);
			float m = n*yVolume.x[i];
			float adl = m_prior + m;
			//System.out.printf("%f %f %f %f %f\n",n,m,adl,EMA3, EMA10);
			m_prior = m;
			EMA3 = (float)( adl*(2.0/11.0)+EMA3*(1-2.0/11.0) );
			EMA10 = (float)( adl*(2.0/51.0)+EMA10*(1-2.0/51.0) );
			x[i] = EMA3-EMA10;
		}		
	}

/*	public float max() {
		// TODO Auto-generated method stub
		return py.max( x );
	}*/

	public void calculateStochasticVolume(Series y, int duration) {
		for( int i=0; i<length(); i++ ) {
			float low = y.x[i]; 
			float high = y.x[i]; 
			if( duration>1 ) {
				for( int d=0; d<=duration; d++ ) {
					if( i>=d && y.x[i-d]<low ) low=y.x[i-d];
					if( i>=d && y.x[i-d]>high ) high=y.x[i-d];
				}
			}
			x[i] = (high != low) ? (y.x[i]-low)/(high-low) :0;
		}
	}

	public Series differential() {
		double[] y = new double[x.length];
		y[0]=0;
		for( int i=1; i<x.length; i++ ) {
			y[i] = x[i]-x[i-1];
		}
		return new Series(y);
	}

	public Series cutOff(double min, double max) {
		double[] y = new double[x.length];
		for( int i=0; i<x.length; i++ ) {
			 
			if( x[i]<min ) y[i]=(float)min;
			else if( x[i]>max ) y[i]=(float)max;
			else y[i] = x[i]; 
		}
		return new Series(y);
	}

	public Series isGreaterThan(float limit) {
		double[] y = new double[x.length];
		for( int i=0; i<x.length; i++ ) {
			y[i] = x[i]>limit ? 1 : 0;
		}
		return new Series(y);
	}

	public Series isLessThan(float limit) {
		double[] y = new double[x.length];
		for( int i=0; i<x.length; i++ ) {
			y[i] = x[i]<limit ? 1 : 0;
		}
		return new Series(y);
	}	


	public void copyInto(double[] vector, int pos, int from, int to) {
	
		for( int i=from; i<to; i++ ) {
			vector[pos++] = x[i];
		}	
		
	}

	public void copyNormalizedInto(double[] vector, int pos, int from, int to) {

		//System.out.println( String.format("pos=%d, f=%d, t=%d", pos, from, to) );
		Series xSub = new Series(to-from);
		for( int i=0; i<(to-from); i++ ) {
			xSub.set(i, this.x[i+from] );
		}
		Series y = xSub.normalize();
		
		for( int i=0; i<y.length(); i++ ) {
			vector[pos++] = y.get(i);
		}	

	}

	public Series isAbove(double d) {
		for( int i=0; i<x.length; i++ ) {
			x[i] = ( x[i]>d ) ? 1 : 0;
		}
		return this;
	}

	public Series isBelow(double d) {
		for( int i=0; i<x.length; i++ ) {
			x[i] = ( x[i]<d ) ? -1 : 0;
		}
		return this;
	}

	public Series trim(int len) {
		if( x.length==len ) return this;
		float[] x1 = x;
		float[] x2 = new float[len];
		for( int i=0; i<len; i++ ) {
			if( i<x.length ){
				x2[i] = x1[i];
			}else{
				x2[i]=0;
			}
		}
		x = x2;
		return this;
	}

	public Series derivation() {
		double[] y = new double[x.length];
		y[0]=0;
		for( int i=1; i<x.length; i++ ) {
			y[i] = x[i-1]!=0 ? (x[i]-x[i-1])/x[i-1] : 0;
		}
		return new Series(y);
	}
	
	
}