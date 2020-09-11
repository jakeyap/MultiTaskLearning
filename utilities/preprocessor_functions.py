#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 13:44:51 2020

@author: jakeyap
"""
import re

def remove_urls(string_in):
    '''
    Removes links that start with HTTP/HTTPS in the tweet
    Parameters
    ----------
    tweet_in : string
        tweet
    Returns
    -------
    tweet_out : string
        cleaned tweet.
    '''
    # \S means any non whitespace characcteer
    #re_object = re.compile('http:\S*|https:\S*|www.\S*')
    re_object = re.compile('http:\S*|https:\S*|www.\S*')
    string_out = re_object.sub(repl='[URL]', string=string_in)
    return string_out

def remove_spaces(string_in):
    '''
    Removes newlines and tabs
    Parameters
    ----------
    tweet_in : string
        tweet
    Returns
    -------
    tweet_out : string
        cleaned tweet.
    '''
    re_object = re.compile('\s')
    string_out = re_object.sub(repl=' ', string=string_in)
    return string_out
