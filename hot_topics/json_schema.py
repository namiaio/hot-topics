# -*- coding: utf-8 -*-

from marshmallow import Schema, fields


class Article(Schema):
    title = fields.Str()
    month = fields.Int()
    year = fields.Int()


class Item(Schema):
    articles = fields.Nested(Article(), many=True)
    topics = fields.List(fields.Str())


class Result(Schema):
    created_at = fields.DateTime()
    clusters = fields.Dict(
        values=fields.Nested(Item()),
        keys=fields.Int()
    )
