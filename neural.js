//Math Library

var matrix = function(rows,cols){
	this.rows = rows;
	this.cols = cols;
	this.data = [];
	var tmp;
	for(var i = 0;i<rows;i++){
		tmp = []
		for(var j = 0;j<cols;j++)
			tmp.push(Math.random());
		this.data.push(tmp);
	}
	this.ones = function(){
		for(var i = 0;i<rows;i++){
			for(var j = 0;j<cols;j++)
				this.data[i][j] = 1;
		}
	};
	this.zeros = function(){
		for(var i = 0;i<this.rows;i++){
			for(var j = 0;j<this.cols;j++)
				this.data[i][j] = 0;
		}
	};
	this.transpose = function(){
		var out = new matrix(this.cols,this.rows);
		for(var i = 0;i<this.rows;i++){
			for(var j = 0;j<this.cols;j++)
				out.data[j][i] = this.data[i][j];
		}	
		return out;
	};
	this.add = function(m){
		var out = new matrix(this.rows,this.cols);
		for(var i = 0;i<this.rows;i++){
			for(var j = 0;j<this.cols;j++)
				out.data[i][j] = this.data[i][j] + m.data[i][j];
		}
		return out;
	};
	this.diff = function(m){
		var out = new matrix(this.rows,this.cols);
		for(var i = 0;i<this.rows;i++){
			for(var j = 0;j<this.cols;j++)
				out.data[i][j] = this.data[i][j] - m.data[i][j];
		}
		return out;
	};
	this.times = function(m){
		var out = new matrix(this.rows,this.cols);
		for(var i = 0;i<this.rows;i++){
			for(var j = 0;j<this.cols;j++)
				out.data[i][j] = this.data[i][j] * m.data[i][j];
		}
		return out;
	};
	this.times_scalar = function(s){
		var out = new matrix(this.rows,this.cols);
		for(var i = 0;i<this.rows;i++){
			for(var j = 0;j<this.cols;j++)
				out.data[i][j] = s*this.data[i][j];
		}
		return out;
	}
	this.dot = function(m){
		var out = new matrix(this.rows,m.cols);
		var s;
		for(var i = 0;i<this.rows;i++){
			for(var j = 0;j<m.cols;j++){
				s = 0
				for(var k = 0;k<this.cols;k++)
					s += this.data[i][k] * m.data[k][j];
				out.data[i][j] = s;
			}
		}
		return out;
	};
	this.eval = function(myfun){
		var out = new matrix(this.rows,this.cols);
		for(var i = 0;i<this.rows;i++){
			for(var j = 0;j<this.cols;j++)
				out.data[i][j] = myfun(this.data[i][j]);
		}
		return out;
	};
	this.from = function(m){
		this.rows = m.length;
		this.cols = m[0].length;
		this.data = m;
	}
	this.from_string = function(s){
		var r = s.split("\n");
		var c;
		var tmp;
		this.rows = r.length;
		this.cols = 0;
		this.data = [];
		for(var i = 0;i<this.rows;i++){
			c = r[i].split(" ");
			tmp = [];
			if(this.cols == 0)
				this.cols = c.length;
			for(var j = 0;j<this.cols;j++)
				tmp.push(parseFloat(c[j]));
			this.data.push(tmp);
		}
	};
	this.to_string = function(){
		var s = "";
		for(var i =0;i<this.rows;i++){
			for(var j = 0;j<this.cols;j++){
				if(j != this.cols-1)
					s += this.data[i][j] + " ";
				else
					s += this.data[i][j];
			}
			s+="\n";
		}
		return s
	};
	this.copy = function(){
		var out = new matrix(this.rows,this.cols);
		for(var i = 0;i<this.rows;i++){
			for(var j = 0;j<this.cols;j++)
				out.data[i][j] = this.data[i][j];
		}
		return out;
	};
}

var sigmoid = function(x){
	return 1/(1 + Math.exp(-x));
}
var dsigmoid = function(x){
	return x*(1-x);
}
var tanh = function(x){
	return (1 - Math.exp(-x))/(1 + Math.exp(-x));
}
var dtanh = function(x){
	return (1 - x**2)/2;
}
var relu = function(x){
	return x>=0?x:0.001*x;
}
var drelu = function(x){
	return x>=0?1:0.001;
}
var linear = function(x){
	return x;
}
var dlinear = function(x){
	return 1;
}

var layer = function(nin,nout,act_fun){
	this.nin = nin;
	this.nout = nout;
	this.act_fun = act_fun;
	this.i = null;
	this.o = null;
	this.d = null;
	this.w = new matrix(nin,nout);
	this.bias = new matrix(1,nout);
	this.w = this.w.times_scalar(1/nin);
	this.bias = this.bias.times_scalar(1/nin);
	this.ones = null;
	this.forward = function(indata){
		this.i = indata;
		var n = this.i.rows;
		this.ones = new matrix(n,1);
		this.o = indata.dot(this.w).add(this.ones.dot(this.bias));
		switch(this.act_fun){
			case "sigmoid":
				this.o = this.o.eval(sigmoid);
				break;
			case "tanh":
				this.o = this.o.eval(tanh);
				break;
			case "relu":
				this.o = this.o.eval(relu);
				break;
			case "linear":
				this.o = this.o.eval(linear);
				break;
			default:
				this.o = this.o.eval(sigmoid);
				break;
		}
		return this.o;
	};
	this.backward = function(errdata){
		switch(this.act_fun){
			case "sigmoid":
				this.d = this.o.eval(dsigmoid).times(errdata);
				break;
			case "tanh":
				this.d = this.o.eval(dtanh).times(errdata);
				break;
			case "relu":
				this.d = this.o.eval(drelu).times(errdata);
				break;
			case "linear":
				this.d = this.o.eval(dlinear).times(errdata);
				break;
			default:
				this.d = this.o.eval(dsigmoid).times(errdata);
				break;
		}
		return this.d.dot(this.w.transpose());
	};
	this.update = function(alpha){
		this.w = this.w.diff(this.i.transpose().dot(this.d).times_scalar(alpha));
		this.bias = this.bias.diff(this.ones.transpose().dot(this.d).times_scalar(alpha));
	};
}

var softmax = function(){
	this.sm = [];
	this.forward = function(indata){
		this.sm = new matrix(indata.rows,indata.cols);
		var s;
		for(var i = 0;i<indata.rows;i++){
			s = 0;
			for(var j = 0;j<indata.cols;j++)
				s += indata.data[i][j];
			for(var j = 0;j<indata.cols;j++)
				this.sm.data[i][j] = indata.data[i][j]/s;
		}
		return this.sm;
	};
	this.backward = function(target){
		var err = this.sm.copy();
		for(var i = 0;i<this.sm.rows;i++){
			err.data[i][target[i]] -= 1;
		}
		return err;
	};
	this.update = function(alpha){

	};
};

var neural_network = function(){
	this.layers = [];
	this.softmax = false;
	this.add_layer = function(nin,nout,act_fun){
		this.layers.push(new layer(nin,nout,act_fun));
	};
	this.add_softmax = function(){
		this.layers.push(new softmax());
		this.softmax = true;
	};
	this.forward = function(indata){
		var tmp = indata;
		for(var i in this.layers){
			tmp = this.layers[i].forward(tmp);
		}
		return tmp;
	}
	this.backward = function(err){
		var tmp = err;
		for(var i = this.layers.length-1;i>=0;i--)
			tmp = this.layers[i].backward(tmp);
	}
	this.update = function(alpha){
		for(var i in this.layers)
			this.layers[i].update(alpha);
	}
	this.train = function(indata,outdata,maxIt,alpha){
		var tmp;
		for(var i = 0 ;i<maxIt;i++){
			tmp = this.forward(indata);
			if(this.softmax){
				this.backward(outdata);
			}
			else{
				tmp = tmp.diff(outdata);
				this.backward(tmp);	
			}
			
			this.update(alpha);
		}
	}
}