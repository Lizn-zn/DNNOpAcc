## Boosting Operational DNN Testing Efficiency through Conditioning

This is a more detailed experimental results for paper `Boosting Operational DNN Testing Efficiency through Conditioning'.

You can easily implement this method by yourself or by modifying this code.

This git is just used for show some experimental results.

#### If we choose different hidden layer rather than the last one, could Cross Entropy method be more effective?

In my opinion, other layers do not enjoy all the following features: 
- High correlation with the prediction accuracy (linear combination only). 
- Independence between neurons facilitating approximation of the joint distribution. The deeper the layer is, the higher the features it encodes, and the less dependent the neurons are. 
- Robustness against the divergence between training and operation data. This is supported by well-known transfer learning practices where only the SoftMax layer is retrained for different tasks. (cf. DOI:10.1109/ICASSP.2013.6639081).
- In addition, the dimension of last layer always lower than others, thus it can decrease the computational cost.

The results also validated this point (the folder `\Layer-selection`). 
In the following figures, the black line is variance of estimation through SRS(Simple Random Sampling); and the red line is vaiance of estimation through CES(Cross Entropy Sampling)

- In driving dataset, we choose last three layer for comparision. 
<table>
    <tr>
        <td ><center><img src="/fig/layer_exp_driving1.png" >The last layer </center></td>
        <td ><center><img src="/fig/layer_exp_driving2.png"  >The penultimate second layer</center></td>
        <td ><center><img src="/fig/layer_exp_driving3.png"  >The penultimate third layer</center></td>

    </tr>
</table>

- 