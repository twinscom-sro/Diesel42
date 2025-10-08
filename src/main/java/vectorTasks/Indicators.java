package vectorTasks;

public class Indicators {

	
	public static Series calculate_willR(Series high, Series low, Series close) {
		Series willR = new Series(close.length());
		for (int i = 0; i < close.length(); i++) {
			int j = i>=14 ? i-13 : 1;
			double hh = high.get(j);
			double ll = low.get(j);
			for( int k=j+1; k<=i; k++ ) {
				if( high.get(k)>hh ) hh = high.get(k);
				if( low.get(k)<ll ) ll = low.get(k);
			}
			willR.set( i,  (float)( 100.0 * (hh-close.get(i)) / (hh-ll) ) );
			//System.out.printf("OBV %d %d %f %f %f\n",j, i,bal,sum,obv.get(i));
		}
		return willR;
	}

	public static Series calculate_obv(Series close, Series volume) {
		Series obv = new Series(volume.length());
		obv.set(0, 0);
		for (int i = 1; i < close.length(); i++) {
			double bal = 0;
			double sum = 0;
			int j = i>=20 ? i-19 : 1;
			for( int k=j; k<=i; k++ ) {
				if( close.get(k)>close.get(k-1) ) {
					bal += volume.get(k);
				}
				if( close.get(k)<close.get(k-1) ) {
					bal -= volume.get(k);
				}
				sum += volume.get(k);
			}
			obv.set( i,  (float)( 100.0 * bal / sum ) );
			//System.out.printf("OBV %d %d %f %f %f\n",j, i,bal,sum,obv.get(i));
		}
		return obv;
	}

	public static Series calculate_xo_pct(Series a, Series b) {
		Series xo = new Series( a.length() );
		for( int i=0; i<xo.length(); i++ ) {
			xo.set(i, (float) ( 100.0*(a.get(i)-b.get(i))/b.get(i) ) ); 
		}
		return xo;
	}

	public static Series calculate_xo_ratio(Series a, Series b, Series c) {
		Series xo = new Series( a.length() );
		for( int i=0; i<xo.length(); i++ ) {
			xo.set(i, (float) ( (a.get(i)-b.get(i))/c.get(i) ) );
		}
		return xo;
	}

	public static Series calculate_xo(Series a, Series b) {
		Series xo = new Series( a.length() );
		for( int i=0; i<xo.length(); i++ ) {
			xo.set(i, (float) ( (a.get(i)-b.get(i))/b.get(i) ) );
		}
		return xo;
	}

	public static Series calculate_pct(Series a, Series b) {
		Series xo = new Series( a.length() );
		for( int i=0; i<xo.length(); i++ ) {
			xo.set(i, (float) ( a.get(i)/b.get(i)*100.0 ) );
		}
		return xo;
	}

	public static Series calculate_pvo(Series volume) {
		Series pvo = new Series(volume.length());
		double shortema = volume.get(0);
		double longema = volume.get(0);
		for (int i = 0; i < pvo.length(); i++) {
			shortema = 0.15*volume.get(i) + 0.85*shortema;
			longema = 0.075*volume.get(i) + 0.925*longema;
			pvo.set(i, (float) ((shortema-longema)/longema * 100.0) );
		}		
		return pvo;
	}

	public static Series calculate_mfi(int days, Series high, Series low, Series close, Series volume) {
			// MFI - Money Flow Index
			Series mfi = new Series(close.length());
			Series positiveMF = new Series(close.length());
			Series negativeMF = new Series(close.length());
			positiveMF.set(0, 0);
			negativeMF.set(0, 0);
			double price_previous = close.get(0);
			for (int i = 1; i < close.length(); i++) {
				double price = ( high.get(i)+low.get(i)+close.get(i) )/ 3.0 ;
				if( price > price_previous ) {
					positiveMF.set( i, (float) (price *  volume.get(i)) );
					negativeMF.set(i, 0);
				}else{
					positiveMF.set( i, 0 );
					negativeMF.set(i, (float) (price *  volume.get(i)) );
				}
				price_previous = price;
			}
			positiveMF.set(0, positiveMF.get(1) );
			negativeMF.set(0, negativeMF.get(1) );	
			
			for (int i = 0; i < mfi.length(); i++) {
				int j = i>=days ? i-days+1 : 0;
				double a=0;
				double b=0;
				for( int k=j; k<=i; k++ ) {
					a += positiveMF.get(k);
					b += negativeMF.get(k);
				}
				//System.out.printf("MFI %d %d  %f %f\n",j,i,a,b);
				if( b!=0 ) mfi.set(i, (float) (100.0 - ( 100.0 / ( 1 + a / b ) ) ) );
				else mfi.set(i, 0);
			}
			return mfi;
	}

	public static Series calculate_macd(Series price ) {
		Series macd = new Series(price.length());
		double shortema = price.get(0);
		double longema = price.get(0);
		for (int i = 0; i < macd.length(); i++) {
			shortema = 0.15*price.get(i) + 0.85*shortema;
			longema = 0.075*price.get(i) + 0.925*longema;
			macd.set(i, (float) ((shortema-longema)/*/longema*/) );
			//System.out.printf("MACD %d %f %f %f\n",i,shortema,longema,macd.get(i));
		}		
		return macd;
	}

	public static Series calculate_macd_pct(Series price ) {
		Series macd = new Series(price.length());
		double shortema = price.get(0);
		double longema = price.get(0);
		for (int i = 0; i < macd.length(); i++) {
			shortema = 0.15*price.get(i) + 0.85*shortema;
			longema = 0.075*price.get(i) + 0.925*longema;
			macd.set(i, (float) (100.0*(shortema-longema)/longema) );
			//System.out.printf("MACD-PCT %d %f %f %f\n",i,shortema,longema,macd.get(i));
		}		
		return macd;
	}

	public static Series calculate_cmf(int days, Series close, Series high, Series low, Series volume) {
		// CMF - chaikin money flow 
		Series cmf = new Series(close.length());
		Series clv = new Series(close.length());
		for (int i = 0; i < clv.length(); i++) {
			double range = (high.get(i) - low.get(i));
			if( range!=0 ) clv.set( i,  (float) ( ( (close.get(i) - low.get(i)) - (high.get(i) - close.get(i)) ) 
					                               / range 
					                           )
					             );
			else clv.set(i, 0);
		}
		for (int i = 0; i < cmf.length(); i++) {
			int j = i>=days ? i-days+1 : 0;
			double a=0;
			double b=0;
			for( int k=j; k<=i; k++ ) {
				a += clv.get(k) * volume.get(k);
				b += volume.get(k);
			}
			//System.out.printf("CMF %d %d  %f %f\n",j,i,a,b);
			if( b!=0 ) cmf.set(i, (float) (a/b) );
			else cmf.set(i, 0);
		}
		return cmf;
	}

	public static Series calculate_ema(int days, Series x) {
		Series ema50 = new Series(x.length());
		double factor = ( 2.0 / (days+1.0));
		float xcur=x.get(0);
		for (int i = 0; i < x.length(); i++) {
			xcur = (float) ( (1-factor)*xcur + factor*x.get(i) );
			ema50.set(i, xcur);
		}
		return ema50;
	}

	public static Series calculate_ma(int days, Series x) {
		Series ma = new Series(x.length());

		for (int i = 0; i < x.length(); i++) {
			int j = i>(days-1) ? i - days + 1 : 0;
			float sum=0;
			for( int k=j; k<=i; k++ ) sum += x.get(k);
			ma.set(i, sum/(i-j+1) );
		}
		return ma;
	}

	public static Series calculate_avg(Series high, Series low, Series close) {
		Series avg = new Series( close.length() );
		for( int i=0; i<close.length(); i++ ) {
			avg.set(i, (float) ( (high.get(i)+close.get(i)+low.get(i))/3.0) ); 
		}
		return avg;
	}

	public static Series calculate_avg(Series high, Series low) {
		Series avg = new Series( high.length() );
		for( int i=0; i<high.length(); i++ ) {
			avg.set(i, (float) ( (high.get(i)+low.get(i))/2.0) ); 
		}
		return avg;
	}

	public static Series dibuch_hh(Series high, int dur) {
		Series dhh = new Series( high.length() );
		for( int i=0; i<high.length(); i++ ) {
			float hh=0;
			for( int j=i-dur; j<i; j++ ) {
				if( j>=0 ) {
					if( hh==0 ) hh=high.get(j);
					else if( hh<high.get(j) ) hh=high.get(j); 
				}
			}
			dhh.set(i, hh ); 
		}
		return dhh;
	}
	
	public static Series dibuch_ll(Series low, int dur) {
		Series dll = new Series( low.length() );
		for( int i=0; i<low.length(); i++ ) {
			float ll=0;
			for( int j=i-dur; j<i; j++ ) {
				if( j>=0 ) {
					if( ll==0 ) ll=low.get(j);
					else if( ll>low.get(j) ) ll=low.get(j); 
				}
			}
			dll.set(i, ll ); 
		}
		return dll;
	}

	public static Series calculate_atr( Series high, Series low, Series close) {
		Series atr = new Series( close.length() );
		double prev_close = close.get(0);
		atr.x[0] = (float)(high.get(0)-low.get(0));
		for( int i=1; i<close.length(); i++ ) {
			double atr1 = Math.abs(high.get(i)-low.get(i));
			double atr2 = Math.abs(high.get(i)-prev_close);
			double atr3 = Math.abs(prev_close-low.get(i));
			if( atr2>atr1 ) atr1 = atr2;
			if( atr3>atr1 ) atr1 = atr3;
			atr.x[i] = (float) atr1;
			prev_close = close.get(i);
		}
		return atr;
	}
	
}
