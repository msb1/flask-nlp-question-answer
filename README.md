<h4>flask-nlp-question-answer</h4>
<h6>Trained question answer model integrated into Flask demo app</h6>
<ol>
<li>Training data set is provided by Google's Natural Questions</li>
<li>Data is cleaned and stored in MongoDB - notebooks: TFQA-bilstm-attn.ipynb</li>
<li>Model is parallel bi-directional lstm for both question and answer text; the second lstm layer is followed by a multihead attention layer</li>
<li>The parallel models are concatenated into dense layers with a sigmoid activation at the final layer</li>
<li>The Spacy large vocabulary model is used for token ID's and vectors for each word</li>
<li>Model is in notebooks: TFQA-bilstm-attn.ipynb</li>
<li>Weights from trained model are saved in file which is too large to upload but can be readily regenerated</li>
<li>A Flask demo app is included with Bootstrap 4 styling:
<ul>
<li>Weights from trained model are imported on demo app startup</li>
<li>Start page has input boxes for question and text corpus.</li>
<li>Text corpus is parsed into potential one, two and three sentence answers which are fed into the trained model</li>
<li>The highest three scores are listed as potential answers with scores in [0, 1)</li>
<li>Named entities are also displayed for the text corpus.</li>
</ul>
</li>
</ol>
