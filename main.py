import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import streamlit as st
import plotly.express as px

st.set_page_config(layout='wide')
ott=st.sidebar.selectbox('OTT Platform',['HOME','NETFLIX','PRIME VIDEO','HOTSTAR'])


def home_page():
       st.sidebar.image(r'netflix.jpg', width=300)
       st.sidebar.image(r'primevideo.jpg', width=300)
       st.sidebar.image(r'hotstar.jpg', width=300)
       st.title('ANALYSIS')
       st.write('The OTT wave gave it further momentum by offering on-demand content, based on individual pre'
                'ferences. External factors like the Covid-19 pandemi'
                'c impacted the streaming wave, making it the most preferred medium of content consumption for viewers across the country.')
       hr11,hr12=st.columns(2)

       subscr=pd.read_csv(r'C:\Users\dagas\OneDrive\Desktop\BE Project Report\Dataset\Netflix_Subscriptions.csv')
       subsciption_chart=px.line(subscr,x='Year',y='Number of subscribers (in millions)',width=600,height=440,text=subscr['Year'])
       subsciption_chart.update_layout(yaxis=dict(showgrid=False),xaxis=dict(showgrid=False))
       subsciption_chart.add_hline(y=203.66,line_color='slategray',line_dash='dash',annotation_text='COVID-19 Outbreak revolutionalize the streaming platforms',annotation_position='top left')
       subsciption_chart.add_hrect(y0=167.09,y1=203.66,line_color='slategray',line_dash='dash')
       #subsciption_chart.add_hrect(y0=202,y1=167)
       subsciption_chart.update_xaxes(showticklabels=False)

       home_csv = pd.read_csv(
              'https://docs.google.com/spreadsheets/d/e/2PACX-1vQGCzz8QVI0Ki7VtlHgFiBoCpUE3g_VJ9KtoiHp8fIyWeLvZIVySi4DWCZPmiyF6DKmo-TFCwYhP4I8/pub?output=csv')
       chart01 = px.bar(home_csv['Where would you go to see the latest movies?'].value_counts().sort_values(),labels={'variable':'Question','index':'Platform','value': 'Count '}, width=600, height=420, )
       chart01.update_layout(xaxis_title=None, yaxis_title=None, xaxis=dict(showgrid=False),
                             yaxis=dict(showgrid=False), )
       chart01.update_traces(showlegend=False)
       # chart01.update_traces(marker=dict(line=dict(color='000000', width=1)))
       hr12.plotly_chart(chart01)
       hr11.plotly_chart(subsciption_chart)
       st.write('The suprising fact is that more people now prefer watching latest movies on Streaming platforms rather than going to a theater ')
       st.write(' ')
       st.write(' ')
       st.title('STREAMING PLATFORM')
       st.write(' ')


       c1,c2,c3=st.columns(3)
       co1,co2=st.columns([1,1])
       cou=home_csv.groupby(['Which OTT platform would you prefer to watch movies and television shows on?'])['Which OTT platform would you prefer to watch movies and television shows on?'].count()

       net=cou['Netflix']
       pri=cou['Amazon Prime Video']
       hots=cou['Hotstar']
       df_len=len(home_csv)
       def num_people(vn):
              count=(((vn*100)/df_len)/10).__ceil__()
              return count
       st.write("The above KPI's shows people preference for different streaming platforms and we can see that Netflix leading the streaming platform"
                "race and the reason is that people feel that Netlfix has a wide varitey of content which is clearly understood from the below graph.")
       c1.info("NETFLIX Preference : "+str(num_people(net))+"/10 people")
       c2.info("AMAZON PRIME Preference : "+str(num_people(pri))+"/10 people")
       c3.info("HOTSTAR Preference : "+str(num_people(hots))+"/10 people")

       chart02 = px.bar(home_csv[
                               'According to you, which platform has a variety of contents?'].value_counts().sort_values(),
                        labels={'value': 'Count '}
                        , orientation='h', width=600, height=300)
       chart02.update_layout(xaxis_title=None, yaxis_title=None, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
       chart02.update_traces(showlegend=False, textfont_size=12, textposition='outside')
       # chart01.update_traces(marker=dict(line=dict(color='000000', width=1)))
       st.plotly_chart(chart02)

       st.write('The suprising fact is that prefer now prefer watching latest movies on Streaming platforms rather than going to a theater ')
       st.write(' ')
       st.write(' ')


       st.title('SUBSCRIPTION RATE')
       srate1,srate2=st.columns([1.5,1])
       subscr_rate_data=pd.read_csv(r'C:\Users\dagas\OneDrive\Desktop\BE Project Report\Dataset\Yearly_SubscriptionFee.csv')
       #srate2.dataframe(subscr_rate_data)
       sub_rate=px.bar(home_csv['Does high subscription rate of one platform, forces you to switch to another platform?'].value_counts().sort_values()
                       ,height=350, width=620,title='Does high Subscription rate of one platform forces you to switch to other platform?')
       sub_rate.update_traces(marker_color=['moccasin','slategray'],showlegend=False)
       sub_rate.update_layout(yaxis=dict(showgrid=False),xaxis_title=None)
       srate1.plotly_chart(sub_rate)
       dd=px.bar(x=subscr_rate_data['Streaming Platform'],y=subscr_rate_data['Rate'],width=520,height=310)
       #srate2.plotly_chart(dd)



       st.title('OTT GENRE')
       aaaaa = st.selectbox(options=['Gender', 'Age'], label='Genre Preference')
       if aaaaa == 'Gender':
              selvalue = 'Your Gender'
       elif aaaaa == 'Age':
              selvalue = 'Your Age Group'
       ott_genre=px.bar(home_csv[selvalue].sort_values(),width=1000,color=home_csv['Genre that you prefer.'])
       ott_genre.update_layout(yaxis=dict(showgrid=False))
       st.plotly_chart(ott_genre)


       st.write('https://docs.google.com/forms/d/e/1FAIpQLSeV47z2_GKXmP8kQTQNGi-0AfEoXsfbuq9b8FXZx5q3pbNTdg/viewform?usp=sf_link')



def netflix():
       st.sidebar.image(r'netflix.jpg', width=300)
       st.title('NETFLIX')
       st.write("Netflix is a subscription-based streaming service that allows our members to watch TV shows and movies without commercials on an internet-connected device. You can also download TV shows and movies to your device and watch without an internet connection. If youre already a member and would like to learn more about using Netflix, visit www.netflix.com")


       netflix_data = pd.read_csv(r'C:\Users\dagas\OneDrive\Desktop\BE Project Report\Dataset\netflixData.csv')
       df_netflix = netflix_data.assign(names=netflix_data['Genres'].str.split(",")).explode('names')
       unique_genre=df_netflix['names'].unique()
       df_netflix['IMDB'] = df_netflix['Imdb Score'].str.split("/")

       df11 = pd.read_csv(r'C:\Users\dagas\OneDrive\Desktop\BE Project Report\Dataset\netflix_titles.csv')
       d1 = df11.groupby(['type', 'Release Year']).size().reset_index(name='Total Content')
       d1 = d1[d1['Release Year'] >= 1990]
       st.write('The below chart displays the total number of movies and televison series that Netlfix had over the period of years.')
       fig3 = px.line(d1, x='Release Year', y='Total Content', color='type',width=1200)
       fig3.update_layout(width=1100,hovermode='x unified')
       fig3.update_xaxes(showgrid=False)
       fig3.update_yaxes(showgrid=False)
       st.plotly_chart(fig3)

       content_type = st.sidebar.radio('Content Type', ['Movie', 'TV Show'])
       col1, col2=st.columns([2,1])



       grouped = netflix_data['Content Type'].value_counts().reset_index()
       piec = px.pie(values=grouped['Content Type'], color_discrete_sequence=px.colors.sequential.RdBu, width=400,
                     hole=.3, hover_name=['Movies','Television Series'])
       piec.update_traces(marker=dict(line=dict(color='#000000', width=4)))
       col2.plotly_chart(piec)

       df_netflix_rating=netflix_data[netflix_data['Content Type']==content_type]
       total_rating=df_netflix_rating.groupby(by='Rating')['Rating'].count().sort_values(ascending=False)
       fig2=px.bar(total_rating,y='Rating',color_discrete_sequence=px.colors.sequential.OrRd)
       fig2.update_layout(xaxis_title='Rating',yaxis_title='Total',height=400,width=700)
       col1.plotly_chart(fig2)








       st.sidebar.text(' ')
       release_unique=df_netflix['Release Date'].unique()
       st.sidebar.write(' ')
       sidebarslider=st.sidebar.slider('Year Wise Top Movies',2009,2021)
       df2=netflix_data[netflix_data['Release Date']==sidebarslider]
       df2=df2[df2['Content Type']==content_type]

       df2=df2.sort_values(by='Imdb Score', ascending=False)
       df2 = df2.reset_index()
       st.sidebar.dataframe(df2[['Title','Imdb Score']].head(5))
       st.markdown("""
       <style>
       body {
         background-color: #3B5992; 
       }
       </style>
           """, unsafe_allow_html=True)

       st.write(' ')
       st.title('RECOMMENDATION SYSTEM')
       option1=st.selectbox('Select the recommendation Type',['Genre Based Recommendation','Movie Description Based Recommendation'])
       with st.expander('Recommendation Models'):
              st.write('1. Genre Base Recommendation: Based on the genre you select, the model provides the Top 10 Movies for that particular movie\n2. Movie Description Based Recommendation: The model calculates a score for the movie description and compares it with the description of other movies')


       st.write(' ')
       cols1, cols2 = st.columns([2, 0.5])
       if option1=='Genre Based Recommendation':
              option = cols1.selectbox('Select Your Genre', unique_genre)
              def genre_recommend(genre):
                     genre_rec = df_netflix[df_netflix['names'] == genre]
                     genre_rec = genre_rec.sort_values(by='Imdb Score', ascending=False)
                     genre_rec = genre_rec.reset_index().head(10)
                     cols1.dataframe(genre_rec[['Title','Imdb Score','Production Country','Duration']])
              genre_recommend(option)


       elif option1=='Movie Description Based Recommendation':
              movie_namee=cols1.text_input('Enter Movie Name')
              def matrix_cosine():
                     df = pd.read_csv(r'C:\Users\dagas\OneDrive\Desktop\BE Project Report\Dataset\netflix_titles.csv')
                     feature1 = ['director', 'cast', 'country', 'description', 'listed_in']
                     for featur in feature1:
                            df[featur] = df[featur].fillna('')

                     def combine_feature(row):
                            return row['director'] + ' ' + row['cast'] + ' ' + row['country'] + ' ' + row[
                                   'description'] + ' ' + row['listed_in']

                     df['combine_feature'] = df.apply(combine_feature, axis=1)
                     cv = CountVectorizer()
                     count_matrix = cv.fit_transform(df['combine_feature'])
                     cosine_sim = cosine_similarity(count_matrix)
                     def title_from_index(df1,index):
                            return df1[df1.index==index]['title'].values[0]

                     def index_from_title(df, title):
                            return df[df['title'] == title].index.values[0]

                     def select_movies(a=movie_namee):
                            movie_index = index_from_title(df, a)
                            similar_movies = list(enumerate(cosine_sim[movie_index]))
                            sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:]

                            i = 0
                            for elements in sorted_similar_movies:
                                   st.write(title_from_index(df, elements[0]))
                                   i = i + 1
                                   if i >= 5:
                                          break
                     select_movies()
              matrix_cosine()

       df_netflix_chart = df_netflix[df_netflix['Content Type'] == content_type]
       total_genre = df_netflix_chart.groupby(by='names')['names'].count().sort_values(ascending=False).head(25)
       fig1 = px.bar(total_genre, y='names', color_discrete_sequence=px.colors.sequential.RdPu)
       fig1.update_layout(yaxis_title='Total', xaxis_title='Genre', height=400, width=1100)

       st.plotly_chart(fig1)

       st.write('https://docs.google.com/forms/d/1fVtLVCPvNZoF7jp8adpu20RYA664Rl22JsrO0Yo6iuw/edit')
       #age = st.slider('Select your age', 1, 100)
       #gender = st.radio('Select your Gender', ['Male', 'Female', 'Other'])
       av = cols2.text_input('Get Movie Information')
       def fetch_poster(movie_id):
              response = requests.get(
                     "https://api.themoviedb.org/3/movie/{}?api_key=8030ff5026927b1c94bda92e986d5deb&language=en-US".format(
                            movie_id))
              data = response.json()
              #pdf=pd.read_json(data)

              image_path = "http://image.tmdb.org/t/p/w500/" + data["poster_path"]
              moviename_path=data['original_title']
              abcd=moviename_path.replace(' ','_')
              google_search="https://en.wikipedia.org/wiki/"+abcd

              cols2.image(image_path,width=150)
              cols2.write(moviename_path)
              cols2.write(google_search)



       fetch_poster(av)
       st.write(
              'https://docs.google.com/forms/d/e/1FAIpQLSeV47z2_GKXmP8kQTQNGi-0AfEoXsfbuq9b8FXZx5q3pbNTdg/viewform?usp=sf_link')




def primevideo():
       st.sidebar.image(r'primevideo.jpg', width=300)


       s1 = st.sidebar.radio('Content Type', ['Movie', 'TV Show'])

       st.title('PRIME VIDEO')
       st.write("Amazon Prime Video is an American subscription video on-demand over-the-top streaming and rental service of Amazon offered as a standalone service or as part of Amazon's Prime subscription. The service primarily distributes films and television series produced by Amazon Studios and MGM Holdings or licensed to Amazon, as Amazon Originals, with the service also hosting content from other providers, content add-ons, live sporting events, and video rental and purchasing services. visit www.primevideo.com")

       df1 = pd.read_csv(r'C:\Users\dagas\OneDrive\Desktop\BE Project Report\Dataset\amazon_prime_titles.csv')
       df2 = pd.read_csv(r'C:\Users\dagas\OneDrive\Desktop\BE Project Report\Dataset\titles.csv')
       dataset = pd.merge(df1, df2, on="title")



       dataset.rename(columns={'type_x': 'type', 'release_year_x': 'year', 'listed_in': 'genre',
                               'description_x': 'description'}, inplace=True)
       final_dataset1 = dataset.drop(
              ['show_id', 'director', "cast", 'date_added', 'duration', 'age_certification', 'production_countries',
               'seasons', 'imdb_id', 'imdb_votes', 'type_y', 'id', 'description_y', 'genres', 'release_year_y',
               'tmdb_popularity', 'tmdb_score'], axis=1)
       final_dataset1.rename(columns={'release_year_x': 'year', 'listed_in': 'genre', 'description_x': 'description'})

       st.sidebar.write(' ')
       sidebarslider=st.sidebar.slider('Year Wise Top Movies',2009,2021)

       abc=dataset[(dataset['type']==s1)&(dataset['year']==sidebarslider)]
       abc=abc.sort_values(by='imdb_score',ascending=False)
       abc=abc[['title','imdb_score']].head()
       st.sidebar.dataframe(abc)

       st.write(' ')
       st.title('RECOMMENDATION SYSTEM')
       option1 = st.selectbox('Select the recommendation Type',
                              ['Genre Based Recommendation', 'Movie Description Based Recommendation'])
       with st.expander('Recommendation Models'):
              st.write(
                     '1. Genre Base Recommendation: Based on the genre you select, the model provides the Top 10 Movies for that particular movie\n2. Movie Description Based Recommendation: The model calculates a score for the movie description and compares it with the description of other movies')

       if option1=='Genre Based Recommendation':
              dataset = dataset.assign(names=dataset['genre'].str.split(",")).explode('names')
              genre_uniuq=dataset['names'].unique()
              genre_val=st.selectbox('Select your preferred genre',genre_uniuq)
              genre_dataframe=dataset[dataset['names']==genre_val]
              genre_dataframe=genre_dataframe[['title','country','imdb_score','duration']].sort_values(by='imdb_score',ascending=False).head(10)
              st.dataframe(genre_dataframe)

       elif option1=='Movie Description Based Recommendation':
              movie_name=st.text_input('Enter Movie Name')
              type_movie = final_dataset1[final_dataset1['type'] == 'Movie']
              type_tv = final_dataset1[final_dataset1['type'] == 'TV Show']
              type_tv = type_tv[['title', 'genre', 'year', 'rating']]
              type_tv = type_tv.sort_values(by=['rating'], ascending=False)
              type_movie = type_movie.sort_values(by=['rating'], ascending=False)
              final_dataset2 = final_dataset1[
                     ['title', 'type', 'country', 'year', 'rating', 'genre', 'description', 'runtime', 'imdb_score']]
              final_dataset2['description'].apply(lambda x: x.lower())

              import nltk
              from nltk.stem.porter import PorterStemmer as ps
              def stem(text):
                     y = []
                     for i in text.split():
                            y.append(ps.stem(i))
                     return " ".join(y)

              from sklearn.feature_extraction.text import CountVectorizer
              cv = CountVectorizer(max_features=5000, stop_words='english')
              vector = cv.fit_transform(final_dataset2['description']).toarray()
              from sklearn.metrics.pairwise import cosine_similarity
              similarity = cosine_similarity(vector)
              final_dataset2 = final_dataset2.set_index('title')
              from sklearn.feature_extraction.text import TfidfVectorizer
              tfidfvec = TfidfVectorizer()
              tfidf_plot = tfidfvec.fit_transform((final_dataset2["description"]))
              from sklearn.metrics.pairwise import cosine_similarity
              cos_sim = cosine_similarity(tfidf_plot, tfidf_plot)

              indices = pd.Series(final_dataset2.index)
              final_dataset2 = final_dataset2.fillna(0)

              from scipy.spatial import distance
              def recommendations(title):
                     recommended_movie = []
                     index = indices[indices == title].index[0]
                     similarity_scores = pd.Series(similarity[index]).sort_values(ascending=False)
                     top_10_movies = list(similarity_scores.iloc[1:11].index)
                     for i in top_10_movies:
                            recommended_movie.append(list(final_dataset2.index)[i])
                     st.write(recommended_movie)
                     return recommended_movie

              recommendations(movie_name)

       st.write(
              'https://docs.google.com/forms/d/e/1FAIpQLSeV47z2_GKXmP8kQTQNGi-0AfEoXsfbuq9b8FXZx5q3pbNTdg/viewform?usp=sf_link')


def hotstar():
       st.sidebar.image(r'hotstar.jpg', width=290)
       st.title('HOTSTAR')
       st.write('Disney+ Hotstar is an Indian brand of subscription video on-demand over-the-top streaming service owned by Novi Digital Entertainment of Disney Star and operated by Disney Media and Entertainment Distribution, both divisions of The Walt Disney Company. visit www.hotstar.com')
       hotstar = pd.read_csv(r'C:\Users\dagas\OneDrive\Desktop\BE Project Report\Dataset\hotstar_dataset.csv')
       hotstar=hotstar.replace(to_replace="NaN",
           value=np.nan)

       def concat(*args):
              strs = [str(arg) for arg in args if not pd.isnull(arg)]
              return ' '.join(strs) if strs else np.nan

       np_concat = np.vectorize(concat)

       hotstar['Title'] = np_concat(hotstar['title'], hotstar['seasons'])
       hotstar.drop(['seasons', 'title', 'episodes', 'index'], axis=1, inplace=True)
       second_column = hotstar.pop('Title')
       hotstar.insert(1, 'Title', second_column)
       hotstar.columns = ['Hotstar_Id', 'Title', 'Description', 'Genre', 'Year', 'Age_rating', 'RunTime', 'Type',
                     'AverageRating', 'NumVotes']
       type_movie = hotstar[hotstar['Type'] == 'movie']
       type_tv = hotstar[hotstar['Type'] == 'tv']

       type_tv.drop(['RunTime'], axis=1, inplace=True)
       type_tv = type_tv[['Title', 'Genre', 'Year', 'AverageRating']]
       type_movie = type_movie[['Title', 'Genre', 'Year', 'AverageRating', 'RunTime']]
       type_tv = type_tv.sort_values(by=['AverageRating'], ascending=False)
       type_movie = type_movie.sort_values(by=['AverageRating'], ascending=False)
       final_data = hotstar[['Hotstar_Id', 'Title', 'Description', 'Genre', 'AverageRating']]
       final_data['Description'].apply(lambda x:x.lower())
       import nltk
       from nltk.stem.porter import PorterStemmer
       ps = PorterStemmer()

       def stem(text):
              y = []

              for i in text.split():
                     y.append(ps.stem(i))

              return " ".join(y)

       from sklearn.feature_extraction.text import CountVectorizer
       cv = CountVectorizer(max_features=5000, stop_words='english')
       vector = cv.fit_transform(final_data['Description']).toarray()
       from sklearn.metrics.pairwise import cosine_similarity
       similarity = cosine_similarity(vector)

       def recommend(movie):
              index = final_data[final_data['Title'] == movie].index[0]
              distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
              for i in distances[1:6]:
                     print(final_data.iloc[i[0]].Title)

       recommend('Cook Like a Chef 1.0')

       st.write(' ')
       st.title('RECOMMENDATION SYSTEM')
       option1 = st.selectbox('Select the recommendation Type',
                              ['Genre Based Recommendation', 'Movie Description Based Recommendation'])
       with st.expander('Recommendation Models'):
              st.write(
                     '1. Genre Base Recommendation: Based on the genre you select, the model provides the Top 10 Movies for that particular movie\n2. Movie Description Based Recommendation: The model calculates a score for the movie description and compares it with the description of other movies')

       st.write(' ')
       columns1, columns2 = st.columns([2, 0.5])
       if option1 == 'Genre Based Recommendation':
              option = columns1.selectbox('Select Your Genre', ['Comedy','Horror'])
              df1=hotstar[hotstar['Genre']==option]
              st.dataframe(hotstar[['Title','Genre','Year']].head(10).reset_index())

       elif option1== 'Movie Description Based Recommendation':
              movie_namee = st.text_input('Enter Movie Name')
              def recommend(movie):
                     index = final_data[final_data['Title'] == movie].index[0]
                     distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
                     for i in distances[1:6]:
                            st.write(final_data.iloc[i[0]].Title)

              recommend(movie_namee)

       content_type = st.sidebar.radio('Content Type', ['movie', 'tv'])
       st.sidebar.write(' ')
       sidebarslider = st.sidebar.slider('Year Wise Top Movies', 2009, 2021)

       hostar_sidebar = hotstar[hotstar['Year'] == sidebarslider]
       hostar_sidebar = hostar_sidebar[hostar_sidebar['Type'] == content_type]

       hostar_sidebar = hostar_sidebar.sort_values(by='AverageRating', ascending=False)
       hostar_sidebar = hostar_sidebar.reset_index()
       st.sidebar.dataframe(hostar_sidebar[['Title', 'AverageRating']].head(5))

       st.write(
              'https://docs.google.com/forms/d/e/1FAIpQLSeV47z2_GKXmP8kQTQNGi-0AfEoXsfbuq9b8FXZx5q3pbNTdg/viewform?usp=sf_link')



if ott=='HOME':
       home_page()
elif ott=='NETFLIX':
       netflix()
elif ott=='PRIME VIDEO':
       primevideo()
elif ott=='HOTSTAR':
       hotstar()