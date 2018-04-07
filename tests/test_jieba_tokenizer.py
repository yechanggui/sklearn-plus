#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `jieba_tokenizer` package."""


import unittest

from sklearn_plus.preprocessing.text.zh import JiebaTokenizer


class TestJieba_tokenizer(unittest.TestCase):
    """Tests for `sklearn_plus` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_basic(self):
        tokenizer = JiebaTokenizer()
        assert tokenizer.transform(['今天上海的天气']) == [u'今天 上海 的 天气']

    def test_words(self):
        tokenizer = JiebaTokenizer(words=['上海的天气'])
        assert tokenizer.transform(['今天上海的天气']) == [u'今天 上海的天气']
