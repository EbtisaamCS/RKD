# 
<h1>Robust Knowledge Distillation in Federated Learning: Counteracting Backdoor Attacks</h1>


<h2>Overview</h2>
    <p>RKD is an effective framework for mitigating challenging backdoor attacks within the FL context, such as A3FL,F3BA, DBA, and TSBA.  Our experiments considered the CIFAR-10, EMNIST, and Fashion-MNIST datasets under various Non-IID conditions and for different attack ratios to evaluate our proposal. RKD demonstrated the capability to maintain high accuracy levels and significantly reduce attack success rates, highlighting its robustness and efficiency capabilities in this context. The paper has been submitted to the SaTML2025 </p>

<h2>Getting Started</h2>
    <ol>
        <li>Clone this repository to your local machine.</li>
        <li>Install the required dependencies using <code>pip</code>:</li>
        <pre><code>pip install -r requirements.txt</code></pre>
        <li>Follow the instructions in the <code>datasetLoaders</code> directory to set up the datasets for training and testing.</li>
        <li>Adjust hyperparameters and settings in the configuration file <code>CustomcConfig.py</code> to suit your experiments</li>
        <li>Start training the RFCL model by running the following command:</li>
        <pre><code>python main.py </code></pre>
    </ol>
   







