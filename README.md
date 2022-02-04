In this post I discuss the preparation of high frequency datasets for Apple (Apple Inc., NASDAQ, ticker: AAPL), JPM (JPMorgan Chase & Co., NYSE, ticker: JPM), and the EURUSD currency pair. I use volatility prediction as my problem setting because the second moment dominates. Careful data cleaning is one of the most important aspects of working with high frequency data and it is not always straightforward to construct a time series of interest from raw tick data. The methods I employ to prepare my data and compute a returns series are therefore are not trivial. I shall use these datasets in several of my subsequent posts, so a thorough exposition of the steps I have taken along with description of the resultant datasets is important. I consider the data from the perspective of the financial econometrician; subsequently I shall contront the data with a variety of deep learning and reinforcement learning methods and a solid understanding of the data will help the interpretation of results. This is a lengthy and detailed post, with many references to supporting research. I hope to show, amongst other things, that simply fitting a time series of asset prices to a deep learning neural network, without proper consideration of the data itself and what it means, is a necessary but not sufficient condition for robust results. Put another way, when we obtain a prediction from our model, how certain can we be that what we observe is signal and not noise.

|___high-frequency-data		
|   |___.env			Do I need a .env file?
|   |___.flaskenv		Is there a jupyter lab equivalent to .flaskenv?
|   |___.git			
|   |___.gitignore		
|   |___.ipynb_checkpoints	excluded from remote via .gitignore
|   |___data/			local data, excluded from remote via .gitignore
|   |___hfdata.ipynb		post, to be served via Voila
|   |___rem.ipynb		workings notebook, excluded from remote via .gitignore
|   |___[i.e., hfdata.py]	python library for this post, NOT REQUIRED FOR THIS POST
|   |___static/			If its style won’t change, it’s a static file rather than a template
|   	|___images		for serving images
|   |___venv/			Separate venv for each post
