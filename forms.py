
import os
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField
from wtforms.validators import DataRequired


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'frosty-bear'
    INIT_FLAG = False
    TEXT = None


class QATextForm(FlaskForm):
    question = StringField('question')
    text = TextAreaField('text')
    submit = SubmitField('Submit')
