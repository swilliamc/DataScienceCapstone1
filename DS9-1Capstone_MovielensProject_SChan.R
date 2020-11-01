##########################################################
# Capstone Movielens Project by Suhaimi William Chan
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Here is the dimension of our edx set
dim(edx)

# Here is the dimension of our validation set
dim(validation)


# Create a training set (90% of edx) and a test_set (10% of edx data)
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)


# Here is the dimension of our train_set
dim(train_set)

# Here is the dimension of our test_set
dim(test_set)

# Now we are doing our data exploration using edx data set, to see a more complete data set, instead of using train set
# We can see some examples of our edx data set with available columns
head(edx)

# We can see classes of our edx data set
str(edx)

# We can see the number of unique movies and users in edx set
edx %>%
  summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

# We can see some examples of top movie genre list in edx set
edx %>% group_by(genres) %>% 
  summarise(n=n()) %>%
  arrange(desc(n)) %>%
  head()

# We can see the quantity of some popular genre movies in edx set
genres = c("Action", "Adventure", "Children", "Comedy", "Drama", "Romance", "Sci-Fi", "Thriller")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

# We can see some of the most rated movies in edx set
edx %>% group_by(movieId, title) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# We can see the rating summary of all movies in edx set
edx %>% group_by(rating) %>% summarize(n=n()) %>% arrange(desc(rating))

# We can visually see the distribution of movie ratings in edx set
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
library(ggthemes)
library(scales)
edx %>%
  group_by(rating) %>%
  summarise(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line() +
  scale_y_continuous(labels = comma) + 
  ggtitle("Movie Rating Distribution", subtitle = "Higher ratings are more common") + 
  xlab("Rating") +
  ylab("Count") +
  theme_economist()

# We can visually see the distribution of number of ratings by number of movies in edx set
edx %>% group_by(movieId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "black") +
  scale_x_log10() + 
  ggtitle("Distribution of Number of Ratings by Number of Movies", 
          subtitle = "The distribution is almost like normal distribution (quite symetric)") +
  xlab("Number of Ratings") +
  ylab("Number of Movies") + 
  theme_economist()

# We can visually see the distribution of number of ratings by number of users in edx set
edx %>% group_by(userId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "black") +
  scale_x_log10() + 
  ggtitle("Distribution of Number of Ratings by Number of Users", 
          subtitle="The distribution is skewed right (positively skewed), the mean is typically greater than the median. ") +
  xlab("Number of Ratings") +
  ylab("Number of Users") + 
  scale_y_continuous(labels = comma) + 
  theme_economist()

# We create our loss function: The residual mean square error (RMSE)
# We decided the best algorithm based on the residual mean squared error (RMSE) on a test set.
# We can interpret the RMSE similarly to a standard deviation: it is the typical error we make when predicting a movie rating. 
# If this number is larger than 1, it means our typical error is larger than one star, which is not good.
# We write a function that computes the RMSE for vectors of ratings and their corresponding predictors:
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# We know that the estimate that minimizes the RMSE is the least squares estimate of mu and, in this case, is the average of all ratings:
mu <- mean(train_set$rating)
mu
#[1] 3.512456

# Our first model formula is Yu,i = mu + Eu,i
# If we predict all unknown ratings with mu, we obtain the following RMSE:
naive_rmse <- RMSE(test_set$rating, mu)
naive_rmse
# [1] 1.060054
# Just the average model RMSE is 1.060054 (our first model)

# We are going to store all our model RMSE in rmse_result to compare to our RMSE goal
options(pillar.sigfig = 7)
rmse_results <- tibble(method = "RMSE Goal", RMSE = 0.86490)
rmse_results
rmse_results <- bind_rows(rmse_results, 
                          tibble(method = "Just the average", 
                                 RMSE = naive_rmse))
rmse_results

# Our second model: modeling movie effect formula will be adding movie bias as follow: Yu,i = mu + b_i + Eu,i
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarise(b_i = mean(rating - mu))

# We can see that these estimates vary substantially:
qplot(b_i, data = movie_avgs, bins = 10, color = I("black"))

# Remember mu = 3.5 so a b_i = 1.5 implies a perfect five star rating.
# Let's see how much our prediction improves once we use yu,i = mu + b_i:
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
Meffect_rmse <- RMSE(predicted_ratings, test_set$rating)
Meffect_rmse
# [1] 0.9429615
# Movie effect model RMSE is 0.9429615 (our second model)
rmse_results <- bind_rows(rmse_results, 
                          tibble(method = "Movie effect model", 
                                 RMSE = Meffect_rmse))
rmse_results


# Let's compute the average rating for user u for those that have rated over 100 movies:
train_set %>% 
  group_by(userId) %>% 
  summarise(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

# Our third model: modeling movie + user effect formula will be adding user bias as follow: Yu,i = mu + b_i + b_u + Eu,i
# We will compute an approximation by computing mu and b_i and estimating b_u as the average of yu,i - mu - b_i:
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

# We can now construct predictors and see how much the RMSE improves using our User effect model:
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
MUeffect_rmse <- RMSE(predicted_ratings, test_set$rating)
MUeffect_rmse
# [1] 0.8646843
# Movie + User Effect model RMSE is 0.8646843 (our third model)
rmse_results <- bind_rows(rmse_results, 
                          tibble(method = "Movie + User Effect model", 
                                 RMSE = MUeffect_rmse))
rmse_results


# Our fourth model: Regularized Movie + User Effect Model
# We can use regularization for the estimate user effects as well. We are minimizing:
# Sum ( (yu,i - mu - bi - bu)^2 + lambda(sum(bi^2) + sum(bu^2)) )
# Choosing the best penalty terms. Lambda is a tuning parameter. We can use cross-validation to choose it.

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

#For the full model, the optimal lambda is:
qplot(lambdas, rmses)
lambda <- lambdas[which.min(rmses)]
lambda
# [1] 5


# Let's compute regularized movie + user effect model
lambda <- 5

mu <- mean(train_set$rating)

b_i <- train_set %>% 
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu)/(n()+lambda))

b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - b_i - mu)/(n()+lambda))

predicted_ratings <- test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

GMUeffect_rmse <- RMSE(predicted_ratings, test_set$rating)
GMUeffect_rmse
# Regularized Movie + User Effect Model RMSE is 0.8641362 (our fourth model)
rmse_results <- bind_rows(rmse_results, 
                          tibble(method = "Regularized Movie + User Effect model", 
                                 RMSE = GMUeffect_rmse))
rmse_results


# Final predicted_ratings using edx and validation set
# Let's compute regularized movie + user effect model
lambda <- 5

mu <- mean(edx$rating)

b_i <- edx %>% 
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu)/(n()+lambda))

b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - b_i - mu)/(n()+lambda))

predicted_ratings <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

GMUeffect_validation_rmse <- RMSE(predicted_ratings, validation$rating)
GMUeffect_validation_rmse
# Regularized Movie + User Effect Model RMSE is 0.8648177 (our fourth model)

# RMSE (25 points): RMSE < 0.86490, so we should get full 25 points as 0.8648177 < 0.86490
rmse_results <- bind_rows(rmse_results, 
                          tibble(method = "Final Regularized Movie + User Effect model (edx vs validation)", 
                                 RMSE = GMUeffect_validation_rmse))
rmse_results
