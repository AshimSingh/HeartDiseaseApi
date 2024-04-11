# Generated by Django 5.0.1 on 2024-04-11 11:46

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='PredictionResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Age', models.IntegerField(default=0)),
                ('Sex', models.IntegerField(default=0)),
                ('Chest_pain', models.IntegerField(default=0)),
                ('Cholesterol', models.IntegerField(default=0)),
                ('Prediction', models.IntegerField(default=0)),
            ],
        ),
    ]
