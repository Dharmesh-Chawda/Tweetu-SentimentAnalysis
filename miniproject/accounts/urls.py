from django.urls import path
from . import views 
from django.contrib.auth.views import LoginView,LogoutView

urlpatterns =  [
    path('',views.indexView,name="home"),
    path('dashboard/',views.dashboardView,name="dashboard"),
    path('login/',LoginView.as_view(),name="login_url"),
    path('register/',views.registerView,name="register_url"),
    path('logout/',LogoutView.as_view(template_name='registration/logout.html'),name="logout"), 
    
    path('home/',views.homeView,name="home"), 
    path('aboutus/',views.aboutView,name="about"), 
    path('results/',views.resultView,name="result"), 
] 
