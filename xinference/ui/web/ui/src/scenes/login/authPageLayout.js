import { Box } from '@mui/material'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import * as React from 'react'
import { Fragment } from 'react'

import Header from './header'

// Shared "guided" shell for the login and setup pages: a pitch + feature
// highlights on the left (hidden on small screens), and a form card on the
// right. Built with plain flexbox rather than MUI's Grid, since Grid's
// negative-margin compensation for item padding overflows the viewport on
// narrow screens unless carefully re-tuned.
export default function AuthPageLayout({
  title,
  description,
  features,
  children,
}) {
  return (
    <Fragment>
      <Header />
      <Box
        sx={{
          minHeight: '100vh',
          width: '100%',
          overflowX: 'hidden',
          boxSizing: 'border-box',
          pt: { xs: 12, md: 16 },
          pb: 8,
          px: { xs: 2, sm: 4 },
          background: (theme) =>
            theme.palette.mode === 'dark'
              ? 'radial-gradient(circle at 15% 20%, rgba(63,81,181,0.18), transparent 55%), radial-gradient(circle at 85% 80%, rgba(0,150,136,0.14), transparent 55%)'
              : 'radial-gradient(circle at 15% 20%, rgba(63,81,181,0.10), transparent 55%), radial-gradient(circle at 85% 80%, rgba(0,150,136,0.08), transparent 55%)',
        }}
      >
        <Box
          sx={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 6,
            width: '100%',
            maxWidth: 1080,
            mx: 'auto',
            alignItems: 'center',
            boxSizing: 'border-box',
          }}
        >
          {/* Left column: pitch + feature highlights (hidden on small screens) */}
          <Box
            sx={{
              display: { xs: 'none', md: 'block' },
              flex: '1 1 420px',
              minWidth: 0,
              maxWidth: '100%',
            }}
          >
            <Typography
              component="h1"
              variant="h3"
              sx={{
                fontWeight: 700,
                background: (theme) =>
                  `linear-gradient(90deg, ${theme.palette.primary.main}, #26a69a)`,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                mb: 2,
              }}
            >
              {title}
            </Typography>
            <Typography
              variant="body1"
              color="text.secondary"
              sx={{ mb: 4, maxWidth: 420 }}
            >
              {description}
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {features.map(
                ({
                  icon: Icon,
                  title: featureTitle,
                  description: featureDescription,
                }) => (
                  <Box key={featureTitle} sx={{ display: 'flex', gap: 2 }}>
                    <Box
                      sx={{
                        flexShrink: 0,
                        width: 40,
                        height: 40,
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        bgcolor: 'primary.main',
                        color: 'primary.contrastText',
                      }}
                    >
                      <Icon fontSize="small" />
                    </Box>
                    <Box>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                        {featureTitle}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {featureDescription}
                      </Typography>
                    </Box>
                  </Box>
                )
              )}
            </Box>
          </Box>

          {/* Right column: the form */}
          <Box sx={{ flex: '1 1 380px', minWidth: 0, maxWidth: '100%' }}>
            <Paper
              elevation={3}
              sx={{
                p: { xs: 3, sm: 4 },
                borderRadius: 3,
                maxWidth: 440,
                mx: 'auto',
              }}
            >
              {children}
            </Paper>
          </Box>
        </Box>
      </Box>
    </Fragment>
  )
}
