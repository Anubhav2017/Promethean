#include <vector>
#include "read.h"
#include "nn.cpp"
using namespace std;

int main(){

    vector<vector <float> > x_train;

    vector<int> y_train;

    read_input(x_train, y_train);
    
    
    Neural_Network nn(10);

    nn.add_conv(32,1,10,1,3,1);
    nn.add_relu(256);
    nn.add_conv(8,32,8,1,3,1);
    nn.add_relu(48);
    nn.add_fcc(48,32);
    nn.add_relu(32);
    nn.add_fcc(32,8);
    nn.add_relu(8);
    nn.add_fcc(8,1);

    
    // nn.add_conv(5,1,28,28,5,5);
    // nn.add_relu(2880);
    // nn.add_fcc(2880,10);
    
    // cout<<'\n';
    
    nn.train_cel(x_train,y_train,0.001,128,10);


}
