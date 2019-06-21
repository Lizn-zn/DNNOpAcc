## Boosting Operational DNN Testing Efficiency through Conditioning

This is a more detailed experimental results for paper `Boosting Operational DNN Testing Efficiency through Conditioning'.

You can easily implement this method by yourself or by modifying this code.

This git is just used for trying to answer the following questions and showing some experimental results.

#### If we choose different hidden layer rather than the last one, could Cross Entropy method be more effective?

In my opinion, other layers do not enjoy all the following features: 

-  Although not necessarily comprehensible for human,  the representation is more stable than the prediction when the operation context is drifting. 

-  The DNN prediction is directly derived from the linear combination of this layer's outputs, and thus it must be highly correlated with the prediction accuracy. 
- Independence between neurons facilitating approximation of the joint distribution. The deeper the layer is, the higher the features it encodes, and the less dependent the neurons are. 
- In addition, the dimension of last layer always lower than others, thus it can decrease the computational cost.

The results also validated this point (the folder `\Layer-selection`). 
In the following figures, the black line is variance of estimation through SRS(Simple Random Sampling); and the red line is vaiance of estimation through CES(Cross Entropy Sampling)

- In MNIST dataset, we choose last 1,3,5 layer for comparison.
<table>
    <tr>
        <td ><center><img src="/fig/layer_exp_mnist1.png" , height=220px><small>The last 1st  layer</small></center></td>
        <td ><center><img src="/fig/layer_exp_mnist2.png" , height=220px><small>The last 3rd layer</small></center></td>
        <td ><center><img src="/fig/layer_exp_mnist3.png" , height=220px ><small>The last 5th layer</small></center></td>
    </tr>
</table>
- In driving dataset, we choose last three layer for comparsion. 
<table>
    <tr>
        <td ><center><img src="/fig/layer_exp_driving1.png" , height=220px><small>The last 1st  layer</small> </center></td>
        <td ><center><img src="/fig/layer_exp_driving2.png"  , height=220px><small>The last 2th layer</small></center></td>
        <td ><center><img src="/fig/layer_exp_driving3.png" , height=220px ><small>The last 3rd layer</small></center></td>
    </tr>
</table>

#### Do any other structural coverage method could be helpful for improving efficiency of operational DNN testing?

The purposes of our work and these structural coverage are clearly different, our work aims to estimate a DNN’s accuracy precisely and efficiently, but not to find individual error-inducing inputs. In addition, we believe these scalar measures are not informative
enough for improving sampling efficiency. To verify this, we experimented
with the Surprise Adequacy in a similar way as CSS, and it turned out to be ineffective. 

I designed 3 operational context as following. 

1. I use the original MNIST training set and normal model, then test it on original MNIST test set. 

2. I use the mutant MNIST training set and mutant model, then test it on original MNIST test set (a similar way in the paper). 

3. I use the transfer dataset for training and testing (the mnist-usps dataset, it can be seen in the folder `SA-exp/mnist-usps`). 

The detailed experimental results are as follows.

<table>
    <tr>
        <td ><center><img src="/fig/SA_exp_original.png" , height=220px><small>The original context</small> </center></td>
        <td ><center><img src="/fig/SA_exp_mutant.png"  , height=220px><small>The mutant context</small></center></td>
        <td ><center><img src="/fig/SA_exp_transfer.png" , height=220px ><small>The transfer context</small></center></td>
    </tr>
</table>


#### What would be the case if the sample size further grows beyond 180? Would different estimators converge?

Due to my limited ability, no very strong theoretical proof has been provided at present. However, i conduct some experiments to validate this point.

I choose the down-sampling dataset (i.e. the context of different system settings in the paper) as test dataset. In this situation, the CES is used for sampling until the size of sample set achieves 2,500. 

<table>
    <tr>
        <td ><center><img src="/fig/large_size_exp.png" , height=400px> </center></td>
            </tr>
</table>

From the figure we can see that the relative effciency is very stable even when the sample size grew up to 2,500.

