
from flask import Flask, request, render_template, redirect, url_for
from forms import QATextForm, Config
from nlp1 import Text

app = Flask(__name__)
app.config.from_object(Config)

@app.route('/', methods=['GET', 'POST'])
def qafxn():
    qaTextForm = QATextForm()
    if not app.config['INIT_FLAG']:
        app.config['TEXT'] = Text()
        app.config['TEXT'].start_model()
        app.config['INIT_FLAG'] = True

    if qaTextForm.validate_on_submit():
        if qaTextForm.submit.data:
            print("Question:", qaTextForm.question.data)
            print("Text:", qaTextForm.text.data)
            text = app.config['TEXT']
            text.tokenize(qaTextForm.question.data, qaTextForm.text.data)
            ner = text.entity_display()
            answer, possibilities = text.answer()
            return render_template('admin/results.html', question=qaTextForm.question.data, text=ner, answer=answer, possibilities=possibilities)

    return render_template('admin/input.html', title='Question Answer', qaTextForm=qaTextForm) # noqa

if __name__ == "__main__":
    app.run()