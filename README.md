<h1>Robust Knowledge Distillation in Federated Learning: Counteracting Backdoor Attacks</h1>

<h2>Overview</h2>
<p>
    RKD is a robust framework designed to mitigate challenging backdoor attacks in federated learning (FL), including A3FL, F3BA, DBA, and TSBA. By combining techniques such as cosine similarity-based clustering, dynamic model selection, knowledge distillation. RKD effectively maintains high Main Task Accuracy (MTA) while significantly reducing the Attack Success Rate (ASR), even under Non-IID conditions.
</p>
<p>
    Our experimental evaluations on CIFAR-10, EMNIST, and Fashion-MNIST datasets—across various attack ratios and degrees of data heterogeneity—demonstrate that RKD consistently outperforms baseline methods. 
</p>
<p>
    This work has been submitted to SaTML2025.
</p>

<h2>Getting Started</h2>
<ol>
    <li>Clone this repository to your local machine.</li>
    <li>Install the required dependencies using <code>pip</code>:
        <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li>Follow the instructions in the <code>datasetLoaders</code> directory to set up the datasets for training and testing.</li>
    <li>Adjust hyperparameters and settings in the configuration file <code>CustomConfig.py</code> to suit your experiments. Options include parameters for data heterogeneity (e.g., \(\alpha\)), attack ratios, and defence parameters such as the minimum cluster size \(Q\).</li>
    <li>Start training the RKD model by running the following command:
        <pre><code>python main.py</code></pre>
    </li>
</ol>
