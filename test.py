t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'discriminator' in var.name]
for var in t_vars: 
    if 'discriminator' in var.name:
        d_vars = [var]
g_vars = [var for var in t_vars if 'generator' in var.name]