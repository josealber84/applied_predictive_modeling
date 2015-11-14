# Join all training data in the data.frame "all.training.data"

library(dplyr)
library(magrittr)
library(lubridate)

setwd("C:/Users/Jose Alberto/Big Data/scripts/applied_predictive_modeling")

# Read data
train <- read.csv("data/train.csv", 
                  stringsAsFactors=FALSE)
store <- read.csv("data/store.csv", 
                  stringsAsFactors=FALSE)
test <- read.csv("data/test.csv", 
                 stringsAsFactors=FALSE)


# Clean and prepare data
all.training.data <-
    train %>%
    mutate(day.week.1 = (DayOfWeek == 1),
           day.week.2 = (DayOfWeek == 2),
           day.week.3 = (DayOfWeek == 3),
           day.week.4 = (DayOfWeek == 4),
           day.week.5 = (DayOfWeek == 5),
           day.week.6 = (DayOfWeek == 6),
           day.week.7 = (DayOfWeek == 7)) %>% 
    select(-DayOfWeek) %>% 
    mutate(Date = ymd(Date),
           week = week(Date),
           month = month(Date),
           year = year(Date)) %>%
    select(-Date, -Customers) %>%
    mutate(state.holiday.0 = (StateHoliday == "0"),
           state.holiday.a = (StateHoliday == "a"),
           state.holiday.b = (StateHoliday == "b"),
           state.holiday.c = (StateHoliday == "c")) %>% 
    select(-StateHoliday) %>%
    left_join(store, by = c("Store")) %>%
    mutate(store.type.a = StoreType == "a",
           store.type.b = StoreType == "b",
           store.type.c = StoreType == "c",
           store.type.d = StoreType == "d") %>% 
    select(-StoreType) %>%
    mutate(assortment.a = Assortment == "a",
           assortment.b = Assortment == "b",
           assortment.c = Assortment == "c") %>%
    select(-Assortment) %>%
    # If there is no competition distance, suppose it is far away
    mutate(CompetitionDistance = ifelse(is.na(CompetitionDistance), 
                                        max(CompetitionDistance, na.rm = T),
                                        CompetitionDistance)) %>%
    mutate(competition.open.months = (2015 - CompetitionOpenSinceYear)*12 +
               7 - CompetitionOpenSinceMonth) %>%
    select(-CompetitionOpenSinceYear, -CompetitionOpenSinceMonth) %>% 
    mutate(promo2.weeks = (2015 - Promo2SinceYear)*52 + 
               31 - Promo2SinceWeek ) %>%
    select(-Promo2SinceWeek, -Promo2SinceYear, -PromoInterval) %>% 
    # Try to help the nets avoid NA efect
    mutate(unknown.competition.open.months = is.na(competition.open.months),
           unknown.promo2.weeks = is.na(promo2.weeks)) %>%
    mutate(competition.open.months = ifelse(is.na(competition.open.months),
                                            0, competition.open.months),
           promo2.weeks = ifelse(is.na(promo2.weeks), 0, promo2.weeks))

# Remove Store column
all.training.data %<>%
    select(-Store)

all.training.data %<>% as.matrix()
