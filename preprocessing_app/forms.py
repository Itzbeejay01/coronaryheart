from django import forms

# Define choices for Yes/No fields
YES_NO_CHOICES = [
    ('Yes', 'Yes'),
    ('No', 'No'),
]

# Define choices for Gender
GENDER_CHOICES = [
    ('Female', 'Female'),
    ('Male', 'Male'),
    # Add other options if necessary for your model/data
]

class SymptomCheckForm(forms.Form):
    age = forms.IntegerField(
        label='Your Age',
        min_value=0,
        max_value=120,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter your age'})
    )
    gender = forms.ChoiceField(
        label='Gender',
        choices=GENDER_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    chest_pain = forms.ChoiceField(
        label='Do you experience chest pain or discomfort?',
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    shortness_of_breath = forms.ChoiceField(
        label='Do you have shortness of breath during activity or at rest?',
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    pain_in_limbs = forms.ChoiceField(
        label='Do you feel pain or discomfort in your arms, neck, jaw, or back?',
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    high_blood_pressure = forms.ChoiceField(
        label='Do you have a history of high blood pressure?',
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    high_cholesterol = forms.ChoiceField(
        label='Do you have high cholesterol levels?',
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    diabetes = forms.ChoiceField(
        label='Do you have diabetes or high blood sugar?',
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    smoker = forms.ChoiceField(
        label='Do you smoke or have a history of smoking?',
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    family_history = forms.ChoiceField(
        label='Do you have a family history of coronary heart disease?',
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    overweight = forms.ChoiceField(
        label='Are you overweight or obese?',
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )

    # Add help text if desired
    age.help_text = "Please enter your current age in years."
    # ... add help text for other fields ...

    def clean_age(self):
        age = self.cleaned_data.get('age')
        if age is None:
            raise forms.ValidationError("Age is required.")
        # Add any other specific age validation if needed
        return age
    
    # Add clean methods for other fields if specific validation is needed beyond choices 