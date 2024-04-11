# Generated by Django 5.0.1 on 2024-04-11 13:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='predictionresult',
            name='Chest_pain',
        ),
        migrations.AddField(
            model_name='predictionresult',
            name='chest_pain_type',
            field=models.IntegerField(db_column='Chest pain type', default=0),
        ),
    ]