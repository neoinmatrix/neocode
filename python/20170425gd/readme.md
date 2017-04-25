<!-- mathjax config similar to math.stackexchange -->
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    jax: ["input/TeX", "output/HTML-CSS"],
    tex2jax: {
        inlineMath: [ ['$', '$'] ],
        displayMath: [ ['$$', '$$']],
        processEscapes: true,
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    },
    messageStyle: "none",
    "HTML-CSS": { preferredFont: "TeX", availableFonts: ["STIX","TeX"] }
});
</script>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>


# <center>batch gradient descent</center> 
$$
J(\theta) = \frac 1 {2m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})^2
$$

$$
\frac{\partial{J(\theta)}}{\partial\theta_j} = \frac 1 m \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})　 x^i_{\theta_j} 
= \frac 1 m   \{  
[( x_{11} \theta_1 + x_{12} \theta_2 + ... + x_{1n} \theta_n) -y_1 ] x_{1j} +  \\
[( x_{21} \theta_1 + x_{22} \theta_2 + ... + x_{2n} \theta_n) -y_2 ] x_{2j}  +  ...  \\
[( x_{m1} \theta_1 + x_{m2} \theta_2 + ... + x_{mn} \theta_n) -y_n ] x_{mj}  \}
$$

$$
\theta_{j}^{'} = \theta_j - \frac{\partial{J(\theta)}}{\partial\theta_j}
$$

$$ \textbf{update} \, \theta_j \quad j=\{1,2,...n\} \, at \, the \, \textbf{same} \, time $$ 

***

# <center>stochastic gradient descent</center> 
$$
J(\theta) = \frac 1 {m} \sum_{i=1}^m  \frac 1 2 (h_\theta(x^{(i)})-y^{(i)})^2 =
\frac 1 {m} \sum_{i=1}^m f(\theta,(x_i,y_i))
$$

$$
\frac{\partial{f(\theta,(x_r,y_r))}}{\partial\theta_j} =  (h_\theta(x^{(r)})-y^{(r)})　 x^r_{\theta_j} 
\quad ( choose \, the \, random \, number \, \textbf r )
$$

$$
\theta_{j}^{'} = \theta_j - \frac{\partial{f(\theta,(x_r,y_r))}}{\partial\theta_j}
$$

$$ \textbf{update} \, \theta_j \quad j=\{1,2,...n\} \, at \, the \, \textbf{same} \, time $$ 
